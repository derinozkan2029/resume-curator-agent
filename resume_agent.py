import os
import subprocess
import requests
import re
import markdown
import json
from datetime import date
from html import escape
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup


from langchain_core.documents import Document
from langchain.chat_models import init_chat_model # importing this for the LLM model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver #for adding memory to the model
from langgraph.types import interrupt,Command 

load_dotenv()

llm =init_chat_model('anthropic:claude-sonnet-4-6')

class ResumeState(TypedDict):
    job_url: str
    job_description: str    # scraped or cleaned text we will get from LinkedIn, HandShake etc. 
    is_suitable: bool | None # to see if qualifications fit 
    fit_reasoning: str | None
    master_resume: str   # the background, loaded once, the second node after starting
    tailored_resume: str # LLM generated text before PDF render
    output_path: str | None
    job_posting: dict | None
    want_cover_letter: bool
    cover_letter: str | None
    cover_letter_path: str | None


class CoverLetterDraft(BaseModel):
    paragraphs: list[str] = Field(..., description="3 or 4 body paragraphs of plain text, with no greeting and no sign-off, 230 to 320 words in total")


COVER_LETTER_RULES = (
    "You write a one-page cover letter for the candidate in the master resume. "
    "Only use facts that are in the master resume. Never invent employers, titles, metrics, dates, or skills. "
    "Describe unfinished work as unfinished, and never state results the master resume does not state. "
    "Keep credit exactly as the master resume gives it (for example 'wrote under the supervision of', not 'co-authored'). "
    "If the job asks for something the resume does not show, do not claim it; leave it out. "
    "The first sentence must be concrete: a specific thing the candidate built or decided, or a specific part of this role. "
    "Never open with 'I am writing to apply' or 'I am excited' or 'I am thrilled'. "
    "Never use the words passionate, thrilled, cutting-edge, groundbreaking, or leverage. "
    "Do not walk through the resume line by line. Pick one or two experiences and tell the short story behind them: "
    "the problem, what the candidate decided, and what happened. "
    "Tie those stories to two or three specifics from the posting, using the posting's own terms only where they are honestly true. "
    "Use short, plain sentences in the first person. Never use em dashes. "
    "End with one specific forward-looking sentence and a brief, sincere thank-you. "
    "Write 3 or 4 paragraphs and 230 to 320 words in total so the letter fits on one page. "
    "Return only the body paragraphs: no greeting, no sign-off, no headings."
)

BANNED_WORDS = ('passionate', 'thrilled', 'cutting-edge', 'groundbreaking', 'leverag')  # 'leverag' also catches leverage, leveraged, leveraging


class FitReasoning(BaseModel):
    is_suitable: bool =Field(..., description= 'Classify whether the master_resume is suitable with the job content'
    'by comparing the required qualifications and actual experience () and checking graduation dates on both the job description and the resume'
    '.')
    fit_reasoning: str = Field(..., description="2-3 sentences on why, citing specific gaps or matches, also classify the gaps as major or minor")


class JobPosting(BaseModel):
    company: str = Field(..., description="Hiring company name")
    title: str = Field(..., description="Job title as written")
    required_skills: list[str] = Field(..., description="Short phrases (max 10), minimum/required qualifications only")
    preferred_skills: list[str] = Field(..., description="Short phrases (max 10), preferred/nice-to-have only")
    start_date: str | None = Field(None, description="Start date as written, or null")
    end_date: str | None = Field(None, description="End date as written, or null")
    availability_requested: bool = Field(..., description="True only if the posting tells candidates to state availability or dates on their resume")
    work_authorization_quote: str | None = Field(None, description="Verbatim sentence about visa sponsorship or work authorization, or null if none")
    sponsorship_restricted: bool = Field(..., description="True if that quote says sponsorship (e.g. H-1B) is unavailable or limited")


def extract_job_posting(state: ResumeState):
    structured_llm = llm.with_structured_output(JobPosting)
    result = structured_llm.invoke([
        {'role': 'system', 'content': 'Extract only what the posting explicitly states. Use null or empty when absent and never guess. Quote the work-authorization line as it is exactly.'},
        {'role': 'user', 'content': state['job_description']},
    ])
    return {'job_posting': result.model_dump()}

def assess_fit(state:ResumeState): #takes a state and does an LLMP prompt with a structured output
    structured_llm = llm.with_structured_output(FitReasoning)

    result = structured_llm.invoke([{'role':'system', 'content': 'Determine/classify whether the job description is suitable with the master resume.'}, 
    {'role': 'user', 'content': f"MASTER RESUME:\n{state['master_resume']}\n\nJOB DESCRIPTION:\n{state['job_description']}"}]) #getting the job description and the master resume
    #this creates an instance of a fit reasoning 
    return {'is_suitable': result.is_suitable, 'fit_reasoning': result.fit_reasoning}


def load_master_resume(path: str):
    text = open(path, encoding='utf-8').read()
    cleaned = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    if len(cleaned)>0:
        return cleaned

def fetch_job_posting(state:ResumeState):
    posting = requests.get(state['job_url'], headers={'User-Agent': '...'})
    posting.raise_for_status() # check the HTML 200
    soup=BeautifulSoup(posting.text, 'html.parser') 
    description_div = soup.find('div', class_='show-more-less-html__markup')
    return {'job_description': description_div.get_text(strip=True) if description_div else ''} #returning a dictionary

def curate_tailored_resume(state: ResumeState):
    messages = [ {'role': 'system', 'content': (
            'Only use experience, skills, and accomplishments present in the master resume; '
            'do not invent qualifications, metrics, or experience not in the source material. '
            'Reorder bullets, reweight which experiences are emphasized, and adjust the professional '
            'summary to fit the job. Mirror the job posting\'s terminology where honestly applicable '
            '(e.g. if the posting says "distributed systems" and the resume has genuinely relevant work, '
            'use that phrasing). Follow the same markdown section headers as the master resume, in the '
            'same order, mirrored exactly, but you may omit an entire section or entry if it is not '
            'relevant to this job (e.g. drop a pre-college/research section for a role that is not '
            'research-focused) rather than only trimming bullets within it. Content must fit strictly '
            'one printed page at ~9.6pt font with 0.45in top/bottom and 0.6in side margins '
            '(roughly 52-56 lines total including headers) -- if a full draft would run longer, cut whole '
            'low-relevance sections/entries first before shortening the sections most relevant to this job. '
            'Preserve every markdown link (e.g. project repo/demo URLs) exactly as written in the master '
            'resume for any entry you keep -- never drop a link from a kept entry to save space. '
            'Omit the "Research (Pre-College)" section entirely unless the job posting is specifically '
            'research-focused (e.g. a research assistantship, PhD-track, or "Student Researcher" role) '
            '-- for general software/technical/business roles it does not differentiate the candidate '
            'and should be cut first when trimming for length. '
            'Format the Certifications section as a single compact line (items separated by " · "), '
            'matching the master resume\'s formatting exactly -- never expand it into a bulleted list, '
            'since that costs several lines for low-differentiation content. For technical/engineering '
            'roles, the project(s) whose technical work most closely matches the posting (e.g. AI/LLM '
            'integration, data pipelines, custom tooling with measured results) should keep 3-4 concrete '
            'technical bullets, not be trimmed to a single generic line -- if space is tight, cut bullets '
            'from a less-relevant entry (e.g. a club/team activity with only high-level, non-technical '
            'detail) before compressing the most technically relevant project down to one bullet. '
            'Never use em dashes (—) -- they read as an AI-writing tell. Use a period, comma, colon, '
            'or semicolon instead, restructuring the sentence if needed.')},
        {'role': 'user', 'content': (  f"MASTER RESUME:\n{state['master_resume']}\n\n" f"JOB DESCRIPTION:\n{state['job_description']}"
        )}, ]
    response = llm.invoke(messages)
    return {'tailored_resume': response.content}

def report_not_suitable(state: ResumeState):
    return {'output_path': None}

def render_pdf(state: ResumeState):
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace', 'templates', 'resume.css') 
    css = open(css_path, encoding='utf-8').read()
    # this builds the path to the CSS file copied earlier, open it, read its raw text into a string.

    # python-markdown only starts a list at a blank line; the master resume (and the LLM output
    # mirroring its structure) puts bullets directly under a link/description line with no blank
    # line between them, so without this they'd get swallowed into the preceding paragraph. Only
    # insert the blank line before the first bullet of a run, not between bullets, so the list
    # stays "tight" (one <li> per line) instead of "loose" (each wrapped in its own <p>).
    lines = state['tailored_resume'].split('\n')
    normalized_lines = []
    for i, line in enumerate(lines):
        is_bullet = line.lstrip().startswith(('- ', '* '))
        prev_line = lines[i - 1] if i > 0 else ''
        prev_is_bullet_or_blank = prev_line.strip() == '' or prev_line.lstrip().startswith(('- ', '* '))
        if is_bullet and not prev_is_bullet_or_blank:
            normalized_lines.append('')
        normalized_lines.append(line)
    markdown_source = '\n'.join(normalized_lines)

    resume_html = markdown.markdown(markdown_source, extensions=['extra', 'sane_lists'])
    # converts the LLM's markdown (headers, bold, links, bullet lists) into real HTML tags
    # so the CSS below actually applies, instead of literal '#'/'**' characters showing up in the PDF.

    full_html = f"""<html>
<head>
<meta charset="utf-8">
<style>
{css}
</style>
</head>
<body>
{resume_html}
</body>
</html>"""

    # an f-string building one complete HTML document as text here.

    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace', 'tmp', 'tailored.html')
    # where we are going to save that HTML string as an actual file, so Chrome can open it.
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    #creates the workspace/tmp/ folder if it doesn't exist yet
    with open(html_path, 'w', encoding='utf-8') as f: #writes the HTML string to disk at html_path
        f.write(full_html)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace', 'output', 'tailored_resume.pdf')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([
        chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={output_path}",
        f"file://{html_path}"
    ], check=True)

    #launches Chrome as a separate program  in headless mode , tells it to load the HTML file we just wrote (file://{html_path}) and print it straight to a PDF at output_path, then exits.

    return {'output_path': output_path}

class CoverLetterDraft(BaseModel):
    paragraphs: list[str] = Field(..., description="3 or 4 body paragraphs of plain text, with no greeting and no sign-off, 230 to 320 words in total")


COVER_LETTER_RULES = (
    "You write a one-page cover letter for the candidate in the master resume. " "Only use facts that are in the master resume. Never invent employers, titles, metrics, dates, or skills. "
    "Describe unfinished work as unfinished, and never state results the master resume does not state. " "Keep credit exactly as the master resume gives it (for example 'wrote under the supervision of', not 'co-authored'). "
    "If the job asks for something the resume does not show, do not claim it; leave it out. "
    "The first sentence must be concrete like a specific thing the candidate built or decided, or a specific part of this role. "
    "Never open with 'I am writing to apply' or 'I am excited' or 'I am thrilled'. " "Never use the words passionate, thrilled, cutting-edge, groundbreaking, or leverage. "
    "Do not walk through the resume line by line. Pick one or two experiences and tell the short story behind them:the problem, what the candidate decided, and what happened. "
    "Tie those stories to two or three specifics from the posting, using the posting's own terms only where they are honestly true. "
    "Use short, plain sentences in the first person. Never use em dashes. " "End with one specific forward-looking sentence and a brief, sincere thank-you. "
    "Write 3 or 4 paragraphs and 230 to 320 words in total so the letter fits on one page. ""Return only the body paragraphs: no greeting, no sign-off, no headings."
)

BANNED_WORDS = ('passionate', 'thrilled', 'cutting-edge', 'groundbreaking', 'leverage', 'leveraged', 'leveraging')  

COVER_LETTER_CSS = """
@page { size: Letter; margin: 0.8in 1in; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", Times, serif; font-size: 11.3pt; line-height: 1.45; color: #1a1a1a; margin: 0; }
h1 { font-size: 16pt; font-weight: 700; margin: 0 0 2px 0; }
h1 .pronouns { font-size: 10.5pt; font-weight: 400; color: #555; }
.contact { font-size: 9.8pt; color: #444; margin-bottom: 20px; }
.contact p { margin: 0; }
.contact a { color: #444; text-decoration: none; }
.date { margin-bottom: 18px; }
p { margin: 0 0 14px 0; }
"""

def check_cover_letter(paragraphs: list[str]) -> list[str]:
    """Return the rules the letter breaks. An empty list means it passes."""
    text = ' '.join(paragraphs)
    problems = []
    if '—' in text:
        problems.append('The letter uses an em dash. So use a period, comma, colon, or semicolon instead.')
    word_count = len(text.split())
    if not 230 <= word_count <= 320:
        problems.append(f'The letter is {word_count} words. It must be 230 to 320 words.')
    if not 3 <= len(paragraphs) <= 4:
        problems.append(f'The letter has {len(paragraphs)} paragraphs. It must have 3 or 4.')
    if paragraphs and re.match(r"\s*I(?: am|['’]m) (?:writing|excited|thrilled)", paragraphs[0], re.IGNORECASE):
        problems.append('The letter opens with a stock phrase. Open with something concrete instead.')
    used = [word for word in BANNED_WORDS if word in text.lower()]

    if used:
        problems.append('The letter uses banned words: ' + ', '.join(used) + '.')
    return problems

def write_cover_letter(state: ResumeState):
    posting = state.get('job_posting') or {}
    details = {key: posting.get(key) for key in ('company', 'title', 'required_skills', 'preferred_skills')}
    structured_llm = llm.with_structured_output(CoverLetterDraft)
    messages = [ {'role': 'system', 'content': COVER_LETTER_RULES},  {'role': 'user', 'content': (
            f"MASTER RESUME (this is the only source of facts):\n{state['master_resume']}\n\n"
            f"TAILORED RESUME (what is being emphasized for this job):\n{state['tailored_resume']}\n\n"
            f"JOB DESCRIPTION:\n{state['job_description']}\n\n" f"JOB DETAILS:\n{json.dumps(details)}\n\n"
            f"FIT NOTES (do not paper over the gaps listed here):\n{state.get('fit_reasoning') or ''}"
        )},
    ]
    paragraphs, problems = [], []
    for attempt in range(3):
        draft = structured_llm.invoke(messages)
        paragraphs= [p.strip() for p in draft.paragraphs if p.strip()]
        problems = check_cover_letter(paragraphs)
        if not problems:
            break
        messages = messages + [ {'role': 'assistant', 'content': '\n\n'.join(paragraphs)}, {'role': 'user', 'content': 'Rewrite the letter and fix these problems:\n- ' + '\n- '.join(problems)},
        ]
    if problems:
        print('Cover letter still breaks these rules after 3 attempts:\n- ' + '\n- '.join(problems))

    return {'cover_letter': '\n\n'.join(paragraphs)}

def print_pdf(html_path: str, pdf_path: str):
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([ chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", f"file://{html_path}"
    ], check=True)

def count_pdf_pages(pdf_path: str) -> int:
    with open(pdf_path, 'rb') as f:
        data = f.read()
    match = re.search(rb'/Type\s*/Pages.*?/Count\s+(\d+)', data, re.DOTALL)

    return int(match.group(1)) if match else 0


def render_cover_letter(state: ResumeState):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    company = (state.get('job_posting') or {}).get('company') or ''
    slug = re.sub(r'[^a-z0-9]+', '_', company.lower()).strip('_') or 'company'

    header_lines = [line.strip() for line in state['tailored_resume'].splitlines() if  line.strip()]
    name_line = header_lines[0].lstrip('#').strip()
    contact_html = markdown.markdown(header_lines[1]) if len(header_lines) > 1 else ''
    match = re.match(r'(.*?)\s*\((.*?)\)$', name_line)
    name, pronouns = (match.group(1), match.group(2)) if match else (name_line, '')
    pronouns_html = f' <span class="pronouns">({escape(pronouns)})</span>' if pronouns else ''
    today = date.today()
    greeting = f'Dear {company} Hiring Team,' if company else 'Dear Hiring Team,'

    body_html = ''.join(f'<p>{escape(p)}</p>' for p in state['cover_letter'].split('\n\n'))

    full_html = f"""<html>
<head>
<meta charset="utf-8">
<style>
{COVER_LETTER_CSS}
</style>
</head>
<body>
<h1>{escape(name)}{pronouns_html}</h1>
<div class="contact">{contact_html}</div>
<div class="date">{today:%B} {today.day}, {today.year}</div>
<p>{escape(greeting)}</p>
{body_html}
<p>Sincerely,<br>{escape(name)}</p>
</body>
</html>"""

    html_path = os.path.join(base_dir, 'workspace', 'tmp', f'cover_letter_{slug}.html')

    pdf_path = os.path.join(base_dir, 'workspace', 'output', f'cover_letter_{slug}.pdf')
    os.makedirs(os.path.dirname(html_path), exist_ok= True)
    os.makedirs(os.path.dirname(pdf_path), exist_ok= True)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print_pdf(html_path, pdf_path)

    with open(pdf_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{greeting}\n\n{state['cover_letter']}\n\nSincerely,\n{name}\n")

    pages = count_pdf_pages(pdf_path)
    if pages!=1:
        print(f'Warning: the cover letter came out {pages} pages; it should be exactly 1.')

    return {'cover_letter_path': pdf_path}






graph_builder = StateGraph(ResumeState)
graph_builder.add_node('assess_fit', assess_fit)
graph_builder.add_node('fetch_job_posting', fetch_job_posting)
graph_builder.add_node('tailor_resume', curate_tailored_resume)
graph_builder.add_node('render_pdf', render_pdf)
graph_builder.add_node('report_not_suitable', report_not_suitable)
graph_builder.add_node('extract_job_posting', extract_job_posting)
graph_builder.add_node('write_cover_letter', write_cover_letter)
graph_builder.add_node('render_cover_letter', render_cover_letter)


graph_builder.add_edge(START, 'fetch_job_posting')
graph_builder.add_edge('fetch_job_posting', 'extract_job_posting') #now this is updated
graph_builder.add_edge('extract_job_posting', 'assess_fit')          
graph_builder.add_conditional_edges('assess_fit', lambda state: 'suitable' if state['is_suitable'] else 'not_suitable', {'suitable': 'tailor_resume', 'not_suitable': 'report_not_suitable'})
# if not suitable we don't need to run it or tailor the resume
graph_builder.add_edge('report_not_suitable', END)
graph_builder.add_edge('tailor_resume', 'render_pdf')
graph_builder.add_conditional_edges('render_pdf', lambda state: 'letter' if state.get('want_cover_letter') else 'done', {'letter': 'write_cover_letter', 'done': END})
graph_builder.add_edge('write_cover_letter', 'render_cover_letter')
graph_builder.add_edge('render_cover_letter', END)
graph_builder.add_edge('render_pdf', END)

graph = graph_builder.compile()

if __name__ == '__main__':
    resume_path = input('Enter path to your resume file (.md or .txt) [default: workspace/resume_master.md]: ').strip()
    if not resume_path:
        resume_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace', 'resume_master.md')
    master_resume = load_master_resume(resume_path)

    job_url = input('Enter job posting URL: ')

    want_letter = input('Also write a cover letter? [y/N]: ').strip().lower().startswith('y')

    initial_state = { 'job_url': job_url, 'job_description': '', 'is_suitable': None, 'fit_reasoning': None,
        'master_resume': master_resume, 'tailored_resume': '', 'output_path': None, 'job_posting': None, 'want_cover_letter': want_letter, 
        'cover_letter': None, 'cover_letter_path': None,
    }
    result = graph.invoke(initial_state)

    if result.get('output_path'):
        print(f"Resume written to {result['output_path']}")
        if result.get('cover_letter_path'):
            print(f"Cover letter written to {result['cover_letter_path']}")
    else:
        print(f"Not a good fit: {result['fit_reasoning']}")