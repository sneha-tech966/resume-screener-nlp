"""
Creates 4 TEST resumes for the authenticity check (data/resumes_pdf_trust_test/):
  resume_11_Rahul_Verma.pdf   weak resume + hidden WHITE text block full of job-description keywords
  resume_12_Pooja_Menon.pdf   weak resume + NEAR-WHITE (#fafafa) text and 1.5 pt micro text with keywords
  resume_13_Sameer_Gupta.pdf  visible keyword stuffing: huge skill list, none of it supported by experience
  resume_14_Kavya_Rao.pdf     HONEST resume with a dark header band and white text (must NOT be flagged)
Run: python tools/make_trust_test_pdfs.py
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

KEYWORDS = ("Python Machine Learning Data Analysis Pandas NumPy Scikit-learn SQL NLP Natural Language Processing Git "
            "TensorFlow Deep Learning Data Science Artificial Intelligence ")
ST_N = ParagraphStyle("n", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#222222"))
ST_H = ParagraphStyle("h", fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#444444"))
ST_S = ParagraphStyle("s", fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=2,
                      textColor=colors.HexColor("#1f4e79"))
ST_T = ParagraphStyle("t", fontName="Helvetica", fontSize=10, leading=13.5)
ST_B = ParagraphStyle("b", parent=ST_T, fontName="Helvetica-Bold", spaceBefore=3)
ST_U = ParagraphStyle("u", parent=ST_T, leftIndent=14, bulletIndent=4)


def section(story, title, lines, bold_first=False):
    story += [Paragraph(title.upper(), ST_S), HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#1f4e79"), spaceAfter=3)]
    for i, l in enumerate(lines):
        if l.startswith("- "):
            story.append(Paragraph(l[2:], ST_U, bulletText="\u2022"))
        else:
            story.append(Paragraph(l, ST_B if (bold_first and i == 0) else ST_T))


def build(path, name, contact, loc, summary, edu, skills, exp, projects, extra=None, band=False):
    story = []
    if band:   # dark header band with WHITE text = legitimate design
        st_wn = ParagraphStyle("wn", parent=ST_N, textColor=colors.white)
        st_wh = ParagraphStyle("wh", parent=ST_H, textColor=colors.white)
        t = Table([[Paragraph(name, st_wn)], [Paragraph(contact, st_wh)], [Paragraph(loc, st_wh)]], colWidths=[174 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1f3a5f")),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
        story += [t, Spacer(1, 6)]
    else:
        story += [Paragraph(name, ST_N), Paragraph(contact, ST_H), Paragraph(loc, ST_H)]
    section(story, "Summary", [summary])
    section(story, "Education", [edu])
    section(story, "Skills", [skills])
    section(story, "Experience", exp, bold_first=True)
    section(story, "Projects", projects)
    if extra:
        story += extra
    SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                      bottomMargin=15 * mm, title=name).build(story)


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "data", "resumes_pdf_trust_test")
os.makedirs(out, exist_ok=True)

# 1. hidden white text block
white = ParagraphStyle("w", parent=ST_T, fontSize=8, textColor=colors.white)
build(os.path.join(out, "resume_11_Rahul_Verma.pdf"), "Rahul Verma",
      "rahul.verma@example.com | +91 90000 12345", "Nashik, Maharashtra",
      "Customer support executive with experience handling calls, tickets and daily reports. Looking for a stable role in a growing company.",
      "B.Com, Pune University, 2019 - 2022",
      "MS Office, Excel, Windows, Customer support, Typing",
      ["Customer Support Executive, Sunrise Infotech, Nashik, Jan 2023 - Dec 2024",
       "- Answered customer calls and resolved complaints within the agreed time.",
       "- Maintained daily call logs in Excel and prepared weekly reports.",
       "- Trained two new team members on the ticketing process."],
      ["- Call Log Tracker: an Excel sheet to track pending customer complaints."],
      extra=[Spacer(1, 6), Paragraph((KEYWORDS * 4).strip(), white)])

# 2. near-white text + micro text
nearwhite = ParagraphStyle("nw", parent=ST_T, fontSize=9, textColor=colors.HexColor("#fafafa"))
micro = ParagraphStyle("mi", parent=ST_T, fontSize=1.5, leading=2, textColor=colors.black)
build(os.path.join(out, "resume_12_Pooja_Menon.pdf"), "Pooja Menon",
      "pooja.menon@example.com | +91 90000 54321", "Mumbai, Maharashtra",
      "Administrative assistant with good communication skills and experience in office coordination and record keeping.",
      "B.A. in English, University of Mumbai, 2018 - 2021",
      "MS Office, Excel, Windows, Communication, Scheduling",
      ["Office Administrator, Bluebird Services, Mumbai, Mar 2022 - Present",
       "- Coordinated meetings, travel bookings and office supplies for a team of twenty.",
       "- Maintained employee records and prepared monthly attendance sheets."],
      ["- Filing System Redesign: reorganised paper and digital records for faster retrieval."],
      extra=[Spacer(1, 6), Paragraph((KEYWORDS * 3).strip(), nearwhite), Paragraph((KEYWORDS * 6).strip(), micro)])

# 3. visible keyword stuffing
build(os.path.join(out, "resume_13_Sameer_Gupta.pdf"), "Sameer Gupta",
      "sameer.gupta@example.com | +91 90000 98765", "Delhi, India",
      "Motivated fresher seeking opportunities in machine learning, data science, NLP and artificial intelligence.",
      "B.Sc. in Mathematics, Delhi College of Arts, 2021 - 2024",
      "Python, Machine Learning, Deep Learning, NLP, Natural Language Processing, Data Science, Data Analysis, Artificial Intelligence, "
      "Computer Vision, Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, Keras, OpenCV, SQL, MySQL, PostgreSQL, MongoDB, AWS, Azure, "
      "Docker, Kubernetes, Git, GitHub, Java, C++, JavaScript, React, Tableau, Power BI",
      ["Sales Associate, Metro Retail Stores, Delhi, Jun 2024 - Present",
       "- Assisted customers on the shop floor and handled billing at the counter.",
       "- Arranged stock on shelves and reported shortages to the store manager."],
      ["- Attendance Register: a simple register to note staff attendance."])

# 4. honest resume with dark header band, white text
build(os.path.join(out, "resume_14_Kavya_Rao.pdf"), "Kavya Rao",
      "kavya.rao@example.com | +91 90000 24680", "Bengaluru, Karnataka",
      "Computer science graduate who builds machine learning models in Python and enjoys exploring data.",
      "B.E. in Computer Science, Bangalore Institute of Technology, 2021 - 2025",
      "Python, Machine Learning, Pandas, NumPy, Scikit-learn, SQL, Git",
      ["Data Science Intern, InsightLoop Technologies, Bengaluru, Jun 2024 - Aug 2024",
       "- Cleaned datasets with Pandas and NumPy and explored them in Python.",
       "- Trained Scikit-learn models for demand forecasting and compared them with cross validation.",
       "- Queried a SQL database to prepare training data and tracked work in Git."],
      ["- Sentiment Analyzer: a Python project that classifies reviews with a machine learning model."],
      band=True)
print("created 4 test PDFs in", out)
