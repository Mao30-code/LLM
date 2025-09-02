# LLM RPA
## Problem
<br>
Each month, the CEO must review the invoicing system and generate a report that consolidates data from various sources. Some of this information is extracted directly from databases, while other information comes from OCR-processed files hosted on a server. This data requires manual analysis to ensure accuracy. However, the finance team struggles to manage the volume of invoices, as the company operates multiple systems, making it difficult to integrate and efficiently process information. <br>
<br>
<br>
Finance Team to provide a full report of pending invoices, coming from a database; on the other hand, there are multiple conversations about their use.
<br>

They are difficult to track and come from different providers, so you're not entirely sure if they've been paid or not.

## Solution
<br>
The automation team proposes an intelligent project combining RPA and LLM. The model will receive invoice data and return a report with information on whether the invoice has been paid or is pending.
<br>

## Architecture
<br>
<br>
1. Fine-tuned AI Model A custom model trained from scratch using previously processed invoices. This model is designed to improve accuracy and automate data interpretation from historical invoice data.
<br>
<br>
2. RPA Workflow with UiPath A robotic process automation (RPA) setup using UiPath that extracts transactions from multiple datasets and processes them through a structured framework, streamlining repetitive tasks.
<br>
<br>
3. LLM Integration with Langchain A large language model (LLM) built with Langchain that generates intelligent prompts based on the processed invoices. It leverages both the fine-tuned model’s data and the RPA outputs to assist in creating the final report. This LLM runs as part of the RPA workflow, ensuring seamless integration and automation throughout the entire reporting process.

## Proposal

This repository includes a script named Fine_tuning.py. Leveraging Hugging Face and the BART base model developed by Facebook AI, I trained a small dataset over three epochs to produce a pre-trained model called invoice_model. This model must be executed prior to initiating the RPA process.
<br>
<br>
As this is a test model, training was conducted using a limited dataset:
<br>
<br>
fac_train.csv
<br>
<br>
fac_validation.csv
<br>
<br>
To run the large language model (LLM), I used a Python script located in the LLM_file.py file. This script loads the previously trained model ./invoice_model. Additionally, a prompt model was created to manage the LLM interaction. The final output includes the invoice record, its category, and the LLM-generated response. This was implemented using Langchain and the google/flan-t5-base model.
<br>
<br>
The next step involves robotic process automation (RPA). In this case, I used UIPath, based on my professional experience. I consistently apply the RE Framework to ensure reliability and structure across all automation models. The framework is divided into the following phases:
<br>
<br>
1. Init (Initialization) This phase loads the project assets. Due to time constraints, only a subset of variables was loaded. The orchestrator and configuration file are used to retrieve these variables.
<br>
<br>
2. Get Transaction Data In this step, the files to be uploaded to the queue are identified. These files represent the transactions that the process will handle. This is the appropriate stage to incorporate various datasets, emails, or PDF documents. For this test, the process reads data directly from a .txt file.
<br>
<br>
3. Process Transaction Here, the LLM is invoked using the transaction data. The Python script is executed, returning results from the pre-trained invoice_model along with a recommendation, both of which are saved in a .txt file. The outputs are stored in a global variable.
<br>
<br>
4. Set Transaction Status This phase manages the status of each transaction.
<br>
<br>
5. End Process The process concludes. At this point, the results are saved to a .txt file. However, a more robust implementation would involve generating a report and sending it via email.

## Why this solution?
<br>
There are several key factors that contribute to the usefulness and superiority of this approach:
<br>
<br>
<b>Efficiency Through Automation</b> By consolidating the entire workload into a fully automated process, significant time savings are achieved. This directly enhances operational efficiency and reduces manual intervention.
<br>
<br>
<b>Adaptability and Value of Output</b> Although the task has limited initial information, the process is highly adaptable to a wide range of datasets. Despite this flexibility, the final output remains consistent and valuable—a comprehensive report that can benefit multiple stakeholders, including the CEO, employees, and the finance department.
<br>
<br>
<b>Scalability and Flexibility</b>  is a core strength of this solution, supported by several factors:
<br>
<br>
The process accommodates various datasets without requiring a fixed structure.
<br>
<br>
The LLM can be fine-tuned and retrained with new invoice data, ensuring continuous improvement.
<br>
<br>
In practical applications, data sources may include databases, email threads, and OCR-enabled PDF invoices.
<br>
<br>
The use of UIPath, along with Assets and Queue management, ensures optimal resource utilization.
<br>
<br>
For deployment across different environments, variables can be easily modified and adjusted, facilitating seamless testing and customization.


## Possible Future Improvements and Extensions

This project was designed as a functional prototype, adaptable to different automation contexts. Listed below are proposed improvements that can be implemented in the short, medium, and long term to scale the system and make it more robust and productive.
<br>
<br>
AI Model and Invoice Classification
<br>
<br>
Scalable Fine-tuning: The BART (Hugging Face) model was trained with small test datasets, but it can be easily scaled with more real invoice data.
<br>
Automatic Vectorization: Currently, vectorization is performed internally; however, chunking has not been applied because the dataset is small. This can be added to improve processing of longer text.
<br>
Softmax classification: The model predicts the invoice category (Pending, Paid, Overdue) using probability scores.
<br>
Improving Model Confidence: Currently, predictions have moderate confidence levels (~80%) due to the limited dataset size. Increasing the training base will significantly improve accuracy.
<br>
Reuse of the trained model: The model runs outside of UiPath for cost and performance reasons, but integrates seamlessly via Python scripts.
<br>
<br>
Integration with UiPath
<br>
<br>
Applied REFramework: The REFramework was used as the basis of the solution to maintain structure, traceability, and consistency.
<br>
Dynamic variables and paths: Paths and files are managed as assets from UiPath Orchestrator to facilitate configurations by environment.
<br>
Security: Secure credentials are planned for email connections (for example, for sending and receiving invoices).
<br>
Email connection: Integration with email servers is planned for automatic download of incoming invoices and sending reports upon completion.
<br>
PDF OCR: Include Document Understanding to extract text from scanned invoices or unstructured PDFs.
<br>
Automatic report generation: Export results to formats such as PDF, Excel, or Power BI for analysis and visualization.
<br>
<br>
Data, Reporting, and Scalability
<br>
<br>
Historical invoice query: Add a database verification script to determine if an invoice has been previously processed (avoiding duplicates).
<br>
Performance metrics: Add tracking of accuracy, error rates, and reliability per batch processed.
<br>
Multi-client orchestration: Design the system to process invoices from multiple clients/companies in parallel using separate queues in UiPath.
<br>
User dashboards: Implement a web or Power BI dashboard so users can view results without entering UiPath.
<br>
## Future Ideas (Out of Scope Prototype)
<br>
Using LangChain with memory: Save conversational context for recurring invoices or frequent suppliers.
<br>
Vector memory with FAISS/Chroma: Semantic search for similar invoices based on text or category.
<br>
Assistant agent per supplier: Configure "mini-agents" by supplier type or category to manage specific tasks (follow-up, alerts, etc.).
<br>
Integration with ERP or CRMs: Connect the solution with tools such as SAP, Odoo, Hubspot, etc.
