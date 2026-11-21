import os
from pathlib import Path
from fpdf import FPDF
from docx import Document

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_pdf(filename: str, title: str, sections: list[tuple[str, str]]):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, body)
        pdf.ln(4)

    pdf.output(str(DATA_DIR / filename))


def create_docx(filename: str, title: str, sections: list[tuple[str, str]]):
    doc = Document()
    doc.add_heading(title, level=1)
    for heading, body in sections:
        doc.add_heading(heading, level=2)
        doc.add_paragraph(body)
    doc.save(str(DATA_DIR / filename))


def create_markdown(filename: str, title: str, sections: list[tuple[str, str]]):
    lines = [f"# {title}\n"]
    for heading, body in sections:
        lines.append(f"## {heading}\n{body}\n")
    (DATA_DIR / filename).write_text("\n".join(lines), encoding="utf-8")


def generate_all():
    # 1. Leave_Policy.pdf
    create_pdf(
        "Leave_Policy.pdf",
        "NexaCore Technologies - Global Leave Policy (2026)",
        [
            ("1. Annual Privilege Leave (PL)",
             "Full-time employees accrue 24 days of Privilege Leave per calendar year (2.0 days per month). "
             "Probationary employees accrue leave but may only avail PL after completing 90 days of continuous service."),
            ("2. Leave Carry-Forward & Encashment",
             "A maximum of 12 unused Privilege Leave days can be carried forward to the next calendar year. "
             "Any balance exceeding 12 days on December 31 lapses automatically unless encashed. "
             "Employees may encash up to 10 days of PL once per year during the Q4 payroll cycle based on Basic Pay."),
            ("3. Sick & Casual Leave (SCL)",
             "Employees receive 10 days of combined Sick and Casual Leave annually. SCL cannot be carried forward "
             "or encashed. Absence exceeding 3 consecutive working days for medical reasons requires a registered "
             "medical practitioner certificate uploaded to the HR portal."),
            ("4. Parental & Bereavement Leave",
             "Primary caregivers are entitled to 26 weeks of paid maternity/adoption leave. Secondary caregivers "
             "receive 4 weeks of paid paternity leave to be utilized within 6 months of childbirth. "
             "Bereavement leave of 5 working days is granted upon the loss of an immediate family member.")
        ]
    )

    # 2. Travel_and_Expense_Policy.pdf
    create_pdf(
        "Travel_and_Expense_Policy.pdf",
        "NexaCore Technologies - Corporate Travel & Expense Policy",
        [
            ("1. Flight & Rail Booking Guidelines",
             "All domestic and international flights must be booked at least 14 days in advance via the NexaTravel portal. "
             "Economy class is mandatory for flights under 6 hours. Premium Economy is permitted for international "
             "flights exceeding 6 hours with VP approval."),
            ("2. Hotel Accommodation Limits",
             "Tier-1 Cities (Bengaluru, Mumbai, Delhi NCR, Hyderabad, Kolkata): Up to INR 8,500 per night excluding taxes. "
             "Tier-2 Cities: Up to INR 5,500 per night. International Metro Cities (London, New York, Singapore): Up to USD 280 per night."),
            ("3. Daily Per-Diem & Meal Allowances",
             "Domestic travel per-diem for meals and local transit is capped at INR 1,800 per day without itemized bills, "
             "or up to INR 2,500 per day with valid GST receipts. Alcohol and personal entertainment are strictly non-reimbursable."),
            ("4. Reimbursement SLA",
             "Expense claims must be submitted within 15 days of trip completion. Approved claims are settled within "
             "7 business days via direct payroll credit.")
        ]
    )

    # 3. IT_Security_and_BYOD_Policy.docx
    create_docx(
        "IT_Security_and_BYOD_Policy.docx",
        "IT Information Security & Acceptable Use Policy",
        [
            ("1. Password & MFA Standards",
             "All corporate accounts require FIDO2 hardware keys or authenticator app MFA. Passwords must be at least "
             "14 characters long and rotated every 180 days. Sharing credentials or API tokens in Slack/Teams is a "
             "Severity-1 security violation."),
            ("2. Bring Your Own Device (BYOD) & VPN",
             "Personal smartphones may access corporate email and Slack only after enrolling in NexaMDM (Mobile Device Management). "
             "Accessing internal production databases, Git repositories, or HR systems from personal laptops is strictly prohibited. "
             "Remote connections must use WireGuard Corporate VPN."),
            ("3. Generative AI Usage Policy",
             "Employees must never paste proprietary source code, customer PII, or unreleased financial figures into "
             "public consumer LLM tools. Only enterprise-approved instances hosted within NexaCore's private VPC are authorized.")
        ]
    )

    # 4. Employee_Handbook_and_Benefits.docx
    create_docx(
        "Employee_Handbook_and_Benefits.docx",
        "Employee Handbook, Hybrid Work & Benefits Guide",
        [
            ("1. Hybrid Work Model",
             "NexaCore operates on a 3-2 hybrid schedule: employees are expected to work from their assigned base office "
             "at least 3 days per week (Tuesday, Wednesday, and Thursday are core collaboration days). Remote work "
             "from outside the employee's home country requires tax-compliance sign-off and is capped at 15 business days/year."),
            ("2. Health Insurance & Wellness",
             "Group Medical Coverage (GMC) provides INR 7,50,000 floater coverage for the employee, spouse, two dependent "
             "children, and dependent parents. Employees also receive an annual wellness stipend of INR 15,000 for "
             "ergonomic home-office gear, gym memberships, or mental health consultations."),
            ("3. Learning & Certification Reimbursement",
             "Employees completing approved cloud, AI, or architecture certifications (e.g., AWS, Azure, IIT executive programs) "
             "receive 100% exam fee reimbursement up to INR 60,000 per fiscal year upon submitting the completion certificate.")
        ]
    )

    # 5. Code_of_Conduct_and_FAQs.md
    create_markdown(
        "Code_of_Conduct_and_FAQs.md",
        "Code of Conduct, Ethics & HR FAQs",
        [
            ("Conflict of Interest & Moonlighting",
             "Full-time employees must not engage in secondary employment, paid consulting, or freelance software development "
             "for external commercial entities without written disclosure and approval from Legal and HR."),
            ("Gift & Hospitality Threshold",
             "Employees may accept nominal promotional gifts from vendors valued below INR 2,500 (or USD 30). Cash gifts "
             "or gift cards of any value are strictly prohibited."),
            ("FAQ: Notice Period & Exit Formalities",
             "The standard notice period for confirmed employees is 60 days (30 days during probation). Buyout of notice "
             "period is subject to project handover sign-off by the Delivery Manager and Business Unit HR."),
            ("FAQ: IT Helpdesk Escalation",
             "For laptop hardware failures or VPN lockouts, raise a P1 ticket at helpdesk.nexacore.internal or call the "
             "24x7 IT bridge at ext. 4400.")
        ]
    )
    print(f"Generated 5 enterprise policy files in {DATA_DIR.resolve()}")


if __name__ == "__main__":
    generate_all()