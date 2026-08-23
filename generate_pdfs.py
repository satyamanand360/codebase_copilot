import os
from markdown_pdf import MarkdownPdf, Section

def create_pdf(md_path, pdf_path):
    print(f"Reading {md_path}...")
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(content))
    pdf.save(pdf_path)
    print(f"Successfully generated {pdf_path}")

def main():
    artifacts_dir = r"C:\Users\satya_matmwh7\.gemini\antigravity-ide\brain\26711838-f707-4cc1-8320-263630d6c348"
    
    interview_md = os.path.join(artifacts_dir, "Interview_Guide.md")
    beginner_md = os.path.join(artifacts_dir, "Beginner_Guide.md")
    
    create_pdf(interview_md, "Codebase_Copilot_Interview_Guide.pdf")
    create_pdf(beginner_md, "Codebase_Copilot_Beginner_Guide.pdf")

if __name__ == "__main__":
    main()
