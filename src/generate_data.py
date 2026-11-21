from pathlib import Path
from fpdf import FPDF
from docx import Document

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_latin1(text: str) -> str:
    """Replaces non-latin1 Unicode punctuation with ASCII equivalents for fpdf2 core fonts."""
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "--",  # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
    }
    for uni_char, ascii_char in replacements.items():
        text = text.replace(uni_char, ascii_char)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def create_pdf(filename: str, title: str, sections: list[tuple[str, str]]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(pdf.epw, 9, _sanitize_latin1(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    for heading, body in sections:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(pdf.epw, 7, _sanitize_latin1(heading), new_x="LMARGIN", new_y="NEXT")

        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(pdf.epw, 6, _sanitize_latin1(body), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    pdf.output(str(DATA_DIR / filename))


def create_docx(filename: str, title: str, sections: list[tuple[str, str]]) -> None:
    doc = Document()
    doc.add_heading(title, level=1)
    for heading, body in sections:
        doc.add_heading(heading, level=2)
        doc.add_paragraph(body)
    doc.save(str(DATA_DIR / filename))


def create_markdown(filename: str, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}\n"]
    for heading, body in sections:
        lines.append(f"## {heading}\n{body}\n")
    (DATA_DIR / filename).write_text("\n".join(lines), encoding="utf-8")


def _generate_pdfs() -> None:
    # 1. Leave_Policy.pdf
    create_pdf(
        "Leave_Policy.pdf",
        "NexaCore Technologies - Global Leave & Holiday Policy (2026)",
        [
            ("1. Annual Privilege Leave (PL)",
             "Full-time employees accrue 24 days of Privilege Leave per calendar year (2.0 days per month, credited on the "
             "1st of each month). Probationary employees accrue PL during probation but may only avail PL after completing "
             "90 days of continuous service. Planned PL exceeding 4 consecutive days requires manager approval at least "
             "14 calendar days in advance."),
            ("2. Leave Carry-Forward & Encashment",
             "A maximum of 12 unused Privilege Leave days can be carried forward to the next calendar year. "
             "Any balance exceeding 12 days on December 31 lapses automatically unless encashed. "
             "Confirmed employees may encash up to 10 days of PL once per year during the Q4 payroll cycle (December), "
             "calculated on Basic Pay (Basic Salary / 26 * encashed days)."),
            ("3. Sick & Casual Leave (SCL)",
             "Employees receive 10 days of combined Sick and Casual Leave annually (pro-rated for mid-year joiners). "
             "SCL cannot be carried forward or encashed at year-end. Absence exceeding 3 consecutive working days for "
             "medical reasons requires a registered medical practitioner certificate uploaded to the NexaHR portal."),
            ("4. Parental, Adoption & Bereavement Leave",
             "Primary caregivers are entitled to 26 weeks of paid maternity or adoption leave (for adopting a child under "
             "1 year of age, 16 weeks of paid leave applies). Secondary caregivers receive 4 weeks of paid paternity leave "
             "to be utilized within 6 months of childbirth or adoption placement. Bereavement leave of 5 working days is "
             "granted upon the loss of an immediate family member (spouse, child, parent, or sibling)."),
            ("5. Compensatory Off (Comp-Off) & Public Holidays",
             "Employees required by their Delivery Manager to work on a scheduled weekend or gazetted holiday earn 1 Comp-Off "
             "credit per full day worked. Comp-Off credits must be utilized within 45 calendar days of accrual or they expire. "
             "NexaCore observes 10 fixed regional public holidays plus 2 floating Optional Holidays per calendar year."),
            ("6. Sabbatical & Higher Education Leave",
             "Employees who have completed 3 consecutive years of full-time service with a minimum 'Exceeds Expectations' "
             "rating in their last two appraisal cycles may apply for an unpaid Sabbatical Leave of 3 to 6 months for "
             "academic research, executive certifications, or personal milestones, subject to BU Head approval.")
        ]
    )

    # 2. Travel_and_Expense_Policy.pdf
    create_pdf(
        "Travel_and_Expense_Policy.pdf",
        "NexaCore Technologies - Corporate Travel & Expense Policy",
        [
            ("1. Flight & Rail Booking Guidelines",
             "All domestic and international flights must be booked at least 14 days in advance via the NexaTravel portal "
             "using the lowest logical airfare within a 3-hour window. Economy class is mandatory for flights under 6 hours. "
             "Premium Economy is permitted for international flights exceeding 6 hours with VP approval. Domestic rail "
             "travel is reimbursed up to AC 2-Tier or Shatabdi/Vande Bharat Executive Chair Car."),
            ("2. Hotel Accommodation Limits",
             "Tier-1 Indian Cities (Bengaluru, Mumbai, Delhi NCR, Hyderabad, Chennai, Kolkata, Pune): Up to INR 8,500 per "
             "night excluding taxes. Tier-2 Cities: Up to INR 5,500 per night. International Metro Cities (London, New York, "
             "San Francisco, Singapore, Tokyo): Up to USD 280 per night. Other International Cities: Up to USD 200 per night."),
            ("3. Daily Per-Diem, Meals & Client Entertainment",
             "Domestic travel per-diem for meals and local transit is capped at INR 1,800 per day without itemized bills, "
             "or up to INR 2,500 per day with valid GST receipts. International per-diem is USD 75 per day. "
             "Client entertainment meals require pre-approval and are capped at INR 4,000 per attendee. Alcohol, tobacco, "
             "in-room movies, and personal laundry (for trips under 4 nights) are strictly non-reimbursable."),
            ("4. Local Cab & Personal Vehicle Mileage",
             "Airport transfers and client-site commutes should use corporate Uber/Ola Business accounts. When using a "
             "personal vehicle for approved client visits, mileage is reimbursed at INR 14 per km for 4-wheelers and "
             "INR 7 per km for 2-wheelers, plus actual toll and parking receipts."),
            ("5. Forex Cards & Reimbursement SLA",
             "For international travel, employees must request a pre-loaded NexaForex multi-currency card at least 7 business "
             "days prior to departure. Expense claims must be submitted in NexaExpense within 15 days of trip completion; "
             "claims submitted after 30 days require CFO exception sign-off. Approved claims settle within 7 business days.")
        ]
    )

    # 3. Performance_and_Appraisal_Policy.pdf (NEW)
    create_pdf(
        "Performance_and_Appraisal_Policy.pdf",
        "NexaCore Technologies - Performance Management & Career Progression Policy",
        [
            ("1. Appraisal Cycles & Rating Scale",
             "NexaCore conducts formal performance reviews biannually: the Annual Appraisal Cycle in April (covering April–March) "
             "and the Mid-Year Check-in & Promotion Cycle in October. Employees are evaluated on a 5-point scale: "
             "5 - Outstanding, 4 - Exceeds Expectations, 3 - Meets Expectations, 2 - Needs Improvement, and 1 - Unsatisfactory."),
            ("2. Eligibility for Annual Merit Increment & Bonus",
             "Employees who join on or before September 30 are eligible for the upcoming April merit increment and variable "
             "performance bonus. Employees joining between October 1 and December 31 receive a pro-rated bonus and are "
             "reviewed in the subsequent cycle. Variable bonus payouts range from 100% (Rating 3) to 140% (Rating 5)."),
            ("3. Internal Job Posting (IJP) & Role Transfers",
             "Confirmed employees must complete at least 12 months in their current project or role and hold a minimum rating "
             "of 'Meets Expectations' (3) before applying for an Internal Job Posting (IJP). Current project release notice "
             "for an approved IJP transfer is capped at 45 calendar days."),
            ("4. Performance Improvement Plan (PIP)",
             "Employees receiving a rating of 1 or 2 are placed on a structured 60-day Performance Improvement Plan (PIP) "
             "co-authored by the Reporting Manager and HR Business Partner with bi-weekly milestone reviews. Successful "
             "completion restores good standing; failure to meet documented KPIs may result in separation.")
        ]
    )


def _generate_docx_files() -> None:
    # 4. IT_Security_and_BYOD_Policy.docx
    create_docx(
        "IT_Security_and_BYOD_Policy.docx",
        "IT Information Security, Cloud & Acceptable Use Policy",
        [
            ("1. Password & Multi-Factor Authentication (MFA)",
             "All corporate SSO, VPN, and cloud accounts require FIDO2 hardware keys or Microsoft/Google Authenticator MFA. "
             "Passwords must be at least 14 characters long (combining uppercase, lowercase, digits, and symbols) and rotated "
             "every 180 days. Sharing credentials, SSH keys, or API tokens in Slack/Teams or public Git repositories is a "
             "Severity-1 security violation subject to immediate disciplinary action."),
            ("2. Bring Your Own Device (BYOD) & Corporate VPN",
             "Personal smartphones (iOS 16+ or Android 13+) may access corporate email and Slack only after enrolling in "
             "NexaMDM (Mobile Device Management) with device encryption and biometric lock enabled. Accessing internal "
             "production databases, Git repositories, or HR systems from personal laptops is strictly prohibited. "
             "All remote connections must route through WireGuard Corporate VPN."),
            ("3. Generative AI & LLM Usage Governance",
             "Employees must never paste proprietary source code, customer PII, credentials, or unreleased financial figures "
             "into public consumer LLM tools (such as public ChatGPT, Claude, or Gemini web interfaces). Only enterprise-approved "
             "LLM instances hosted within NexaCore's private VPC with zero data-retention agreements are authorized."),
            ("4. Removable Media & Hardware Incident SLA",
             "USB flash drives and external hard disks are blocked by default via CrowdStrike Endpoint DLP. Temporary USB "
             "write access requires CISO approval. If a corporate laptop or mobile device is lost or stolen, the employee "
             "must notify SecOps at secops@nexacore.internal within 2 hours so remote wipe can be executed immediately."),
            ("5. Patch Management & Shadow IT Prohibition",
             "Critical OS and browser security patches pushed via NexaMDM/Jamf/Intune must be rebooted and applied within "
             "72 hours of release. Purchasing or connecting third-party SaaS tools to corporate SSO without an InfoSec Vendor "
             "Risk Assessment (VRA) is prohibited.")
        ]
    )

    # 5. Employee_Handbook_and_Benefits.docx
    create_docx(
        "Employee_Handbook_and_Benefits.docx",
        "Employee Handbook, Hybrid Work & Total Rewards Guide",
        [
            ("1. Hybrid Work Model & Core Hours",
             "NexaCore operates on a 3-2 hybrid schedule: employees are expected to work from their assigned base office "
             "at least 3 days per week (Tuesday, Wednesday, and Thursday are designated core collaboration days). "
             "Standard core collaboration hours across Indian delivery centers are 11:00 AM to 4:00 PM IST. Remote work "
             "from outside the employee's home country requires cross-border tax sign-off and is capped at 15 business days/year."),
            ("2. Health Insurance, Checkups & Wellness Stipend",
             "Group Medical Coverage (GMC) provides INR 7,50,000 family floater coverage for the employee, spouse, two dependent "
             "children, and dependent parents (with an optional top-up up to INR 15,00,000). Employees and spouses also receive "
             "one complimentary annual preventive health checkup and an annual wellness stipend of INR 15,000 for ergonomic "
             "home-office furniture, gym memberships, or certified mental health consultations."),
            ("3. Learning, Certifications & Conference Reimbursement",
             "Employees completing approved cloud, AI, data engineering, or architecture certifications (e.g., AWS, Azure, GCP, "
             "IIT executive programs, CKA) receive 100% exam/tuition fee reimbursement up to INR 60,000 per fiscal year upon "
             "submitting the passing certificate and GST invoice, subject to a 12-month retention clawback clause."),
            ("4. Broadband Internet & Relocation Allowance",
             "All full-time employees receive a flat monthly broadband and mobile connectivity allowance of INR 1,500 paid "
             "automatically through payroll. Company-initiated inter-city relocations include up to 15 days of guest-house "
             "accommodation, household goods transit reimbursement, and a one-time settling-in allowance of INR 50,000."),
            ("5. Employee Referral Bonus Program",
             "Employees referring successful candidates hired into open roles earn a referral bonus paid after the new hire "
             "completes 90 days of service: INR 35,000 for Band L2/L3 engineers, INR 55,000 for Band L4 Senior Architects/Leads, "
             "and INR 75,000 for niche AI/Security specialist roles.")
        ]
    )

    # 6. Remote_Work_and_Ergonomics_Policy.docx (NEW)
    create_docx(
        "Remote_Work_and_Ergonomics_Policy.docx",
        "Remote Work, Home Office & Ergonomics Standard Operating Procedure",
        [
            ("1. Home Office Ergonomic Setup Reimbursement",
             "New full-time hires and confirmed employees who have not claimed equipment in the past 24 months may claim up to "
             "INR 18,000 (drawn from or in addition to initial onboarding setup) for an ergonomic mesh chair, external 24-inch+ "
             "monitor, mechanical keyboard, or UPS power backup. Assets reimbursed under this policy remain company property "
             "for 24 months."),
            ("2. Workspace Connectivity & Power Backup Standards",
             "Employees working remotely must maintain a primary broadband connection of at least 100 Mbps download / 25 Mbps "
             "upload speed and a minimum 2-hour power backup for router and laptop operation during core collaboration hours."),
            ("3. Temporary Domestic & International Workations",
             "Employees may work from another Indian city where NexaCore has no office for up to 30 cumulative days per year "
             "with Reporting Manager approval. International workations are restricted to pre-approved countries, capped at "
             "15 business days per calendar year, and require HR + Global Mobility approval 21 days in advance.")
        ]
    )


def _generate_markdown_files() -> None:
    # 7. Code_of_Conduct_and_FAQs.md
    create_markdown(
        "Code_of_Conduct_and_FAQs.md",
        "Code of Conduct, Ethics, Compliance & HR FAQs",
        [
            ("1. Conflict of Interest & Moonlighting Prohibition",
             "Full-time employees must dedicate their primary professional efforts to NexaCore. Secondary employment, "
             "paid consulting, advisory board seats, or freelance software development for external commercial entities "
             "is strictly prohibited without prior written disclosure and approval from Legal and HR."),
            ("2. Gift, Hospitality & Anti-Bribery Thresholds",
             "Under NexaCore's Anti-Bribery and Anti-Corruption (ABAC) policy, employees may accept nominal promotional gifts "
             "from vendors valued below INR 2,500 (or USD 30) once per year. Cash gifts, gift cards, loans, or lavish "
             "hospitality of any value are strictly prohibited and must be declined and reported to Compliance."),
            ("3. Whistleblower & POSH Redressal Channels",
             "NexaCore maintains a zero-tolerance policy toward workplace harassment and ethical violations. Complaints under "
             "the Prevention of Sexual Harassment (POSH) Act can be filed confidentially at posh@nexacore.internal, and "
             "anonymous ethics/whistleblower reports can be submitted 24x7 via ethicsline.nexacore.internal without fear of retaliation."),
            ("4. Social Media & External Speaking Policy",
             "Employees speaking at technical conferences, publishing open-source libraries, or posting about NexaCore on "
             "LinkedIn/X must include a personal-views disclaimer and must never disclose non-public architecture diagrams, "
             "client names, or internal metrics without Corporate Communications approval."),
            ("5. FAQ: Notice Period, Buyout & Exit Formalities",
             "The standard notice period for confirmed employees is 60 days (30 days during probation). Notice period buyout "
             "or early release is discretionary and requires formal project handover sign-off from the Delivery Manager and "
             "Business Unit HR. Full and Final (FnF) settlement is processed within 30 days of the last working day."),
            ("6. FAQ: Payroll Cutoffs & IT Helpdesk Escalation",
             "Monthly attendance regularization and investment/tax proof declarations close on the 20th of every month; "
             "salaries are credited on the last working day of the month. For laptop hardware failures, MFA resets, or VPN "
             "lockouts, raise a P1 ticket at helpdesk.nexacore.internal or call the 24x7 IT bridge at ext. 4400.")
        ]
    )

    # 8. Procurement_and_Vendor_Onboarding_Policy.md (NEW)
    create_markdown(
        "Procurement_and_Vendor_Onboarding_Policy.md",
        "Enterprise Procurement, Software Licensing & Vendor Onboarding Policy",
        [
            ("1. Purchase Order (PO) Approval Thresholds",
             "All third-party software, cloud subscriptions, and hardware purchases require an approved Purchase Requisition "
             "(PR) in NexaProcure before vendor engagement. Purchases up to INR 1,00,000 require Department Head approval; "
             "purchases between INR 1,00,001 and INR 10,00,000 require VP + Finance Controller approval and at least 3 "
             "competitive vendor quotes; purchases exceeding INR 10,00,000 require CFO sign-off."),
            ("2. Mandatory NDA & Master Services Agreement (MSA)",
             "No proprietary specifications or datasets may be shared with a prospective vendor until a bilateral Non-Disclosure "
             "Agreement (NDA) is executed via DocuSign. Standard Legal review SLA is 5 business days for NexaCore paper "
             "and 10 business days for third-party vendor paper."),
            ("3. Vendor Payment Terms",
             "NexaCore's standard vendor payment cycle is Net-45 days from receipt of a valid GST-compliant invoice and "
             "signed Goods/Service Receipt Note (GRN). Advance payments exceeding 20% of contract value require CFO exception approval.")
        ]
    )


def generate_all() -> None:
    _generate_pdfs()
    _generate_docx_files()
    _generate_markdown_files()
    print(f"Generated 8 comprehensive enterprise policy files in {DATA_DIR.resolve()}")


if __name__ == "__main__":
    generate_all()