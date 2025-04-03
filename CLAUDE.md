# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Run server: `uvicorn app.main:app --reload`
- Format code: `black .`
- Sort imports: `isort .`
- Lint: `flake8 .`
- Run tests: `pytest`
- Run single test: `pytest path/to/test_file.py::test_function`

## Code Style Guidelines

- **Formatting**: Use Black formatter with default settings
- **Imports**: Use isort with sections (stdlib, third-party, local)
- **Type Hints**: Include type annotations for function parameters and returns
- **Database Models**: Follow SQLAlchemy conventions with explicit column types
- **API Schemas**: Use Pydantic models for request/response validation
- **Error Handling**: Use try/finally patterns with proper cleanup
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **API Route Structure**: Group related endpoints in routers with consistent tagging

### intial project thought and inception

heres every info that i want claude to get used to about the code in this repo , its everything about the inital design and thought of the code design

> height="1.483853893263342in"}

## UNIVERSITY OF NAIROBI {#university-of-nairobi .unnumbered}

> **FACULTY OF SCIENCE AND TECHNOLOGY DEPARTMENT OF COMPUTING AND
> INFORMATICS**

MEDICAL DIAGNOSTICS ASSISTANT

## BY: {#by .unnumbered}

> **OKELLO JOHN OTIENO**
>
> **SCS3/2286/2023**
>
> A project report submitted in partial fulfillment of the requirements
> for the award of Bachelor of Science in Computer Science of the
> University of Nairobi.

**February 2025**

# DECLARATION {#declaration .unnumbered}

> **ABSTRACT**

# TABLE OF CONTENTS {#table-of-contents .unnumbered}

> [**DECLARATION** **2**](#declaration)
>
> [**ABSTRACT** **3**](#_heading=h.30j0zll)
>
> [**TABLE OF CONTENTS** **4**](#table-of-contents)
>
> [**CHAPTER 1: INTRODUCTION** **5**](#chapter-1-introduction)

1.  [Background 5](#background)

2.  [Problem Statement 6](#problem-statement)

3.  [Objectives 6](#objectives)

4.  [Scope 7](#scope)

> [**CHAPTER 2:** **REVIEW OF SIMILAR SYSTEMS** > **9**](#_heading=h.4d34og8)
>
> 2.1 Review of similar systems [9](#_heading=h.2s8eyo1)
>
> 2.2 Analysis of current gaps in the existing technologies
>
> [**CHAPTER 3: SYSTEMS ANALYSIS AND DESIGN** > **13**](#chapter-3-systems-analysis-and-design)

1.  [System Development Methodology 13](#_heading=h.lnxbz9)

2.  [System Analysis 13](#system-analysis)

    1.  [Requirements Elicitation 13](#requirements-elicitation)

    2.  [Requirements Analysis 15](#requirements-analysis)

3.  [System Design 16](#system-design)

    1.  [Architectural Design 16](#architectural-design)

    2.  [Process Design 17](#process-design)

    ```{=html}
    <!-- -->
    ```

    4.  [Database Design 19](#database-design)

    5.  [Use Case Diagram 20](#use-case-diagram)

4.  [Resources 21](#resources)

5.  [Project Schedule and Gantt Charts 22](#_heading=h.1ci93xb)

> [**CHAPTER 4: SYSTEMS IMPLEMENTATION** > **23**](#chapter-4-systems-implementation)
>
> [**REFERENCES** **24**](#references)

# CHAPTER 1: INTRODUCTION {#chapter-1-introduction .unnumbered}

## Background

Healthcare systems in Kenya, particularly in Level 2 and Level 3
facilities, face significant challenges related to **diagnostic
capacity, medical record-keeping, and clinical decision support
([[https://www.medex.health/en/blog/healthcare-kenya-challenges-and-opportunities]{.underline}](https://www.medex.health/en/blog/healthcare-kenya-challenges-and-opportunities)
)** . Clinicians often work under resource constraints, with limited
access to specialized knowledge and diagnostic tools. The increasing
patient load, combined with the complexity of medical diagnoses, creates
a need for **technological support systems** that can assist clinicians
in their daily practice.

Modern artificial intelligence and **natural language processing (NLP)**
technologies have opened new possibilities for creating supportive tools
that enhance clinical practice without replacing human judgment. Systems
like Ada Health have demonstrated the potential of **AI-assisted
diagnostics**, but there remains a need for solutions specifically
designed for the **Kenyan healthcare context**, including support for
both **English and Swahili**.

This project is particularly **personal** to me because I once suffered
from a **serious medical condition** that was **initially
misdiagnosed**. As a result, my condition worsened, leading to
significantly **higher medical costs** than if it had been correctly
diagnosed earlier. This experience made me realize the **critical
importance of accurate diagnostic tools**, particularly in healthcare
facilities where access to specialists is limited. Inspired by this, I
decided to develop a **Medical Diagnostics Assistant** to help improve
**diagnostic accuracy, medical record-keeping, and clinical decision
support**

## Problem Statement

> Clinicians in Level 2 and Level 3 hospitals in Kenya face significant
> diagnostic challenges due to high patient volumes, limited access to
> specialized medical knowledge, and inefficient medical record-keeping.
> Existing diagnostic tools, like Ada Health and Babylon Health, are not
> optimized for the Kenyan context, lacking Swahili language support and
> offline functionality, which are critical for local healthcare
> facilities. Moreover, patients often struggle to access their medical
> histories, leading to repeated tests, increased costs, and delayed
> diagnoses. A Medical Diagnostics Assistant that supports English and
> Swahili, enhances diagnostic accuracy through AI, provides
> comprehensive medical record-keeping, and operates effectively in
> low-resource settings is essential to improve clinical decision-making
> and patient care continuity.

## Objectives

The primary objective of this project is to develop a Medical
Diagnostics Assistant that enhances clinical decision-making and medical
record-keeping for Level 2 and Level 3 hospitals in Kenya. The system
will integrate AI-based diagnostics and multilingual support to improve
accessibility and accuracy in healthcare.

#### Specific Objectives {#specific-objectives .unnumbered}

1.  To develop a clinical decision support system that assists

    > clinicians in diagnosing patients based on symptoms and medical
    > history.

2.  To implement Natural Language Processing (NLP) capabilities that

    > enable medical record-keeping in English and Swahili for improved
    > accessibility.

3.  To create a knowledge base of common medical conditions, symptoms,

    > and recommended diagnostic approaches relevant to the Kenyan
    > healthcare context.

4.  To design an intuitive user interface that seamlessly integrates

    > into clinical workflows, ensuring ease of use for healthcare
    > providers.\
    > To develop a secure and structured data storage system for patient
    > records that adheres to healthcare data protection standards.

5.  To implement an open-source technology stack that ensures

    > sustainability, adaptability, and future scalability of the
    > system.

6.  To evaluate the system\'s accuracy and usability through controlled
    > testing with healthcare professionals.

## Scope

This project focuses on developing a Medical Diagnostics Assistant
designed primarily for Level 2 and Level 3 hospitals in Kenya. The
system will provide clinical decision support, medical record-keeping,
and symptom-based diagnostic suggestions using AI and Natural Language
Processing (NLP).

**Features Included in the System:**

1.  Symptom-based diagnostic support for common medical conditions in

    > Kenya

2.  Multilingual support (English and Swahili) to improve accessibility

    > for clinicians and patients. Medical record-keeping system for
    > both doctors and patients, ensuring continuity of care.

3.  User management with different access levels for clinicians and

    > administrators.

4.  Secure patient data storage following healthcare data protection

    > regulations.

5.  A responsive web-based interface for accessibility on both computers
    > and mobile devices.

- **Technological Limitations:**

  - The NLP components may face challenges in achieving high

    > accuracy in both English and Swahili due to the complexity of
    > medical terminology.

  - The system relies on a stable internet connection for accessing

    > certain AI and NLP functionalities, which may be a limitation
    > in areas with poor connectivity.

  - The decision support system will require regular updates to the
    > knowledge base, which might be resource-intensive.

## Justification

This project addresses critical needs in the Kenyan healthcare system,
particularly for Level 2 hospitals where resources and specialized
knowledge may be limited. The Medical Diagnostics Assistant will support
improved clinical outcomes through:

1.  Enhanced diagnostic accuracy by providing evidence-based suggestions

    > based on presented symptoms

2.  Improved efficiency in medical record-keeping through natural

    > language processing

3.  Better documentation of patient encounters in languages familiar to

    > both clinicians and patients

4.  Knowledge transfer and educational support for less experienced

    > clinicians

5.  Consistency in diagnostic approaches across different healthcare

    > facilities

6.  Potential for data collection that could inform public health
    > interventions and resource allocation

> By focusing on an open-source technology stack, the project ensures
> sustainability and adaptability, allowing for future improvements and
> extensions as healthcare needs evolve.ysis into public complaints

## CHAPTER 2: REVIEW OF SIMILAR SYSTEMS {#chapter-2-review-of-similar-systems .unnumbered}

### 2.1 Ada Health {#ada-health .unnumbered}

Ada Health ([[https://ada.com/]{.underline}](https://ada.com/) ) is one
of the most prominent AI-driven diagnostic support systems globally.
Launched in 2016, Ada provides a symptom assessment tool that uses a
sophisticated AI engine to suggest possible conditions based on reported
symptoms. The system employs a conversational interface that guides
users through a series of questions tailored to their initial symptoms.

Key features:

- Symptom assessment through conversational interface

- Personalized health assessments based on demographic information and

  > medical history

- Integration capabilities with healthcare provider systems

- Available as both consumer-facing and professional versions

- Multilingual support (though limited in African languages)

Limitations in the Kenyan context:

- Limited optimization for conditions and presentations common in

  > Kenya

- Insufficient support for Swahili and other local languages

- Designed primarily for high-resource healthcare settings

- Limited offline functionality for areas with poor connectivity

```{=html}
<!-- -->
```

-

### 2.2 Medic Mobile (now Medic) {#medic-mobile-now-medic .unnumbered}

Medic Mobile
([[https://medic.org/?%2Fdeploy-medic-mobile]{.underline}](https://medic.org/?%2Fdeploy-medic-mobile)
), now known simply as Medic, has developed an open-source platform for
community health workers that includes decision support tools. While not
primarily focused on diagnostics, their CHT (Community Health Toolkit)
offers valuable insights into developing healthcare technology for
resource-constrained settings.

Key features:

- Mobile-first design for community health workers

- Offline-first functionality for areas with limited connectivity

- Decision support for common community health issues

- Integration with SMS and basic mobile devices

- Open-source approach allowing for customization

Limitations:

- Not specifically designed for clinical diagnostics in hospital

  > settings

- Limited natural language processing capabilities

- Focus on community health rather than clinical decision support

### 2.4 Isabel Healthcare {#isabel-healthcare .unnumbered}

Isabel (
[[https://www.isabelhealthcare.com/]{.underline}](https://www.isabelhealthcare.com/)
) is a diagnosis decision support system designed specifically for
healthcare professionals. It uses pattern recognition software to help
clinicians consider possible diagnoses based on entered symptoms, signs,
and patient demographics.

Key features:

- Extensive database of diseases and conditions

- Integration with electronic health records

- Evidence-based recommendations

- Education-focused approach with links to medical literature

- Designed specifically for healthcare professional use

Limitations in the Kenyan context:

- High cost structure limiting accessibility

- Limited optimization for conditions prevalent in Kenya

- No support for Swahili or other local languages

- Limited offline functionality

### 2.6 Analysis of Gaps in Existing Systems {#analysis-of-gaps-in-existing-systems .unnumbered}

From reviewing these similar systems, several gaps become apparent that
our Medical Diagnostics Assistant aims to address:

1.  Language Support: None of the major systems offer robust support for

    > Swahili and other languages commonly used in Kenyan healthcare
    > settings.

2.  Contextual Relevance: Existing systems are not specifically

    > optimized for the disease patterns, resource constraints, and
    > healthcare practices in Kenyan Level 2 hospitals.

3.  Connectivity Considerations: Many systems require consistent

    > internet connectivity, which may not be available in all Kenyan
    > healthcare facilities.

4.  Integrated Record-Keeping: While some systems offer diagnostic

    > support and others focus on record-keeping, few integrate both
    > functions effectively for the Kenyan context.

5.  Cost and Sustainability: Many commercial systems have cost

    > structures that make them inaccessible for widespread adoption in
    > the Kenyan healthcare system.

6.  Open-Source Approach: Few systems embrace a fully open-source
    > approach that would allow for local adaptation and sustainability.

Our proposed Medical Diagnostics Assistant aims to address these gaps by
creating a solution specifically designed for the Kenyan healthcare
context, with robust language support, contextually relevant diagnostic
assistance, and an open-source architecture that ensures sustainability
and adaptability.

# CHAPTER 3: SYSTEMS ANALYSIS AND DESIGN {#chapter-3-systems-analysis-and-design .unnumbered}

### 3.1 System Development Methodology {#system-development-methodology .unnumbered}

After careful consideration of various software development
methodologies, this project will adopt the Spiral Model for its
development approach. The Spiral Model combines elements of iterative
development with systematic planning and risk management, making it
particularly suitable for this healthcare application where both
innovation and reliability are critical.

Rationale for Selecting the Spiral Model:

1.  Risk Management: The Spiral Model emphasizes early identification

    > and mitigation of risks, which is essential for a healthcare
    > application where patient safety is paramount.

2.  Iterative Development: The model allows for multiple iterations,

    > enabling progressive refinement of the system based on clinical
    > feedback and real-world testing.

3.  Prototype Development: Early prototyping will help validate the

    > system\'s usability and clinical utility before committing to
    > full-scale implementation.

4.  Flexibility: The model accommodates changes in requirements that may
    > emerge as clinicians interact with early versions of the system.

The implementation of the spiral model will involve six main development
phases for each iteration which include requirement elicitation ,
analysis , design , implementation and testing , customer evaluation ,
customer communication .This approach will allow for continuous
refinement of the system while managing the risks associated with
developing a clinical decision support tool.

## System Analysis

## Requirements Elicitation

The process of gathering requirements for the Medical Diagnostics
Assistant involved a combination of techniques to ensure a comprehensive
understanding of both functional and non-functional requirements.
Initial requirements were identified firsthand, and further
investigation was conducted to refine and expand upon these
requirements.

**Interviews:** Interviews were conducted with key stakeholders,
including clinicians, administrators, and medical experts, to uncover
specific challenges faced in clinical diagnostics and medical
record-keeping within Level 2 and Level 3 hospitals in Kenya.
Stakeholders expressed their needs for a system that enhances diagnostic
accuracy, supports multilingual communication, and improves the
efficiency of medical record management.

**Questionnaires:** Structured questionnaires were developed and
distributed to gather detailed information from stakeholders regarding
their specific needs and preferences. These questionnaires included both
closed-ended questions for quantitative data and open-ended questions
for qualitative insights on topics such as the challenges with current
diagnostic practices, the importance of multilingual support, and
desired features of the Medical Diagnostics Assistant.

**Use Cases:** Use cases were developed to illustrate how different user
roles (clinicians, administrators, patients) would interact with the
system. These scenarios provided clarity on the specific actions users
would perform, helping to identify functional requirements related to
user authentication, diagnostic support, medical record management,
knowledge base utilization, and reporting.

## Requirements Analysis

> With the outlined requirements, the proposed system has the following
> functionalities.

### 3.2.1 Functional Requirements {#functional-requirements .unnumbered}

The system must fulfill the following functional requirements:

**User Authentication & Authorization**

1.  Secure login for different user roles (doctors, patients,

    > administrators).

2.  Role-based access control to protect sensitive medical data

**Diagnostic Support**

1.  Doctors input patient symptoms into the system.

2.  The AI-based engine suggests possible diagnoses with confidence

    > scores.

3.  The system recommends additional tests where necessary.

**Medical Record Management**

1.  Allows both clinicians and patients to access medical history.

2.  Supports structured data entry and free-text notes in English &

    > Swahili.

3.  Enables retrieval of past diagnoses for better continuity of care.

**Knowledge Base**

1.  Maintains a database of diseases, symptoms, and recommended

    > diagnostic approaches.

2.  Allows updates as medical knowledge evolves.

**Reporting & Analytics**

1.  Generates reports on common diagnoses, patient visits, and usage

    > patterns.

2.  Tracks system effectiveness in improving clinical decision-making.

### 3.2.2 Non-Functional Requirements {#non-functional-requirements .unnumbered}

1.  **Performance:** The system must return diagnostic results \*\*within

    > 5 seconds\*\*.

2.  **Reliability:** Must maintain **99.5% uptime** during clinical

    > hours.

3.  **Usability:** Designed for ease of use, requiring \*\*minimal

    > training\*\*.

4.  **Security:** Must comply with **healthcare data protection laws**.

5.  **Scalability:** Should accommodate \*\*increasing numbers of users

    > and records\*\*.

6.  **Multilingual Support:** Must process medical terminology in
    > **English and Swahili**.

## System Design

## Architectural Design

![](media/image4.jpg){width="6.739583333333333in"
height="7.1849715660542435in"}

> _Figure 1 : Architectural Design_

## Process Design

> **Data Flow Diagram**

**figure 2 as shown below is representation of the level 1 dfd of the
system , highlighting the flow of data through the system from the start
where the patient / doctor logs in to the point of receipt of a
diagnosis**

![](media/image2.jpg){width="6.291666666666667in"
height="7.598958880139983in"}

> _Figure 2 : data flow diagram : level 1_

## Database Design

The database in the figure below is a key insight in understanding the
organisation of data in the system , the diagram provides a visual
representation of the database for reference

> ![](media/image6.jpg){width="6.59375in" height="4.407385170603675in"}

## f*igure 3 : database design diagram* {#figure-3-database-design-diagram .unnumbered}

## Use Case Diagram

> The various ways in which the user interact with the system is well
> defined bellow in the usecase diagram

![](media/image3.png){width="6.594799868766404in"
height="4.694444444444445in"}

> _Figure 4: Use Case Diagram._

## Technology

Since the system will be consisting of different technologies that need
to be integrated efficiently with each other , while still giving
consideration to the cost of development , the open source ai stack will
be used in the development of the project

#### Frontend {#frontend .unnumbered}

1.  Next.js (React Framework) -- For building an interactive and

    > responsive UI.

2.  Standard CSS (instead of Tailwind) -- For custom styling.

#### Backend {#backend .unnumbered}

1.  FastAPI (Python) -- For handling API requests and system logic.

2.  PyTorch (AI Engine) -- For medical diagnosis using machine learning

    > models.

3.  Hugging Face API -- For NLP tasks like symptom processing and
    > medical note structuring.

#### Database & Caching {#database-caching .unnumbered}

1.  PostgreSQL -- Stores structured medical records.

2.  pgBouncer -- Optimizes database connections (instead of Redis).

#### Deployment {#deployment .unnumbered}

1.  Docker (for backend & database containers).

2.  Nginx (for serving the frontend efficiently).

## Resources

> Below are the resources required to implement the proposed system.
>
> _Software:_ Windows operating system, Visual studio code for code
> editing , git for version control and a browser ( eg chrome or firefox
> )
>
> _Hardware:_ A computer with a minimum of 16GB RAM and i7 processor,
> capable of running small and mid sized machine learning model ,
> development tools and software.
>
> _Time: enough time for all the different phases of the project from
> planning to the deployment and testing ._

**3.4 PROJECT SCHEDULE AND GRANT CHARTS**

**PROJECT SCHEDULE**

---

**Task **Task Name\*\* **Planned Start **Planned End **Deliverables**
No** Date** Date\*\*

---

**1** **Project **15/03/2025\*\* **31/03/2025** **Requirements
Planning** Document\*\*

**2** **System Design** **01/04/2025** **15/04/2025** **Architecture &
Database Schema**

**3** **Knowledge Base **16/04/2025\*\* **20/05/2025** **Medical Knowledge
Development** Base\*\*

**4** **Frontend **21/05/2025\*\* **20/06/2025** **UI Prototype**
Development\*\*

**5** **Backend **21/05/2025\*\* **20/06/2025** **Core API Services**
Development\*\*

**6** **NLP & Diagnostic **21/06/2025\*\* **15/08/2025** **Language Processing
Engine** & Diagnostic Module\*\*

**7** **System **16/08/2025\*\* **31/08/2025** **Integration Test
Integration & Report**
Testing\*\*

**8** **User Testing** **01/09/2025** **15/09/2025** **User Feedback
Report**

**9** **Refinements** **16/09/2025** **05/10/2025** **Updated System**

**10** **Documentation** **06/10/2025** **20/10/2025** **User & Technical
Manual**

**11** **Final **21/10/2025\*\* **31/10/2025** **Deployed System**
Deployment\*\*

**12** **Project Report** **01/11/2025** **15/11/2025** **Final Project
Report**

---

# CHAPTER 4: SYSTEMS IMPLEMENTATION {#chapter-4-systems-implementation .unnumbered}

1.  Resources

    1.  Hardware

    2.  Software

2.  Technologies

# CHAPTER 5: CONCLUSION AND RECOMMENDATION {#chapter-5-conclusion-and-recommendation .unnumbered}

1.  Achievements

2.  Challenges and Limitations

3.  Conclusion

# REFERENCES {#references .unnumbered}

1.  **Academic and Research Sources**

Artificial Intelligence in Medical Diagnostics:

1.  Topol, E. J. (2019). High-performance medicine: the convergence of

    > human and artificial intelligence. Nature Medicine, 25(1), 44-56.

2.  Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M.,

    > Blau, H. M., & Thrun, S. (2017). Dermatologist-level
    > classification of skin cancer with deep neural networks. Nature,
    > 542(7639), 115-118.

3.  Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in
    > clinical practice. Nature Medicine, 25(1), 30-36.

Digital Health Platforms:

1.  Winkler, R., & Söllner, M. (2018). Unleashing the potential of

    > chatbots in education: A state-of-the-art analysis. In Proceedings
    > of the 2018 AIS SIGED International Conference on Information
    > Systems.

2.  Nebeker, C., Torous, J., & Bartlett Ellis, R. (2019). Building the
    > case for actionable ethics in digital health research supported by
    > artificial intelligence. BMC Medicine, 17(1), 139.

Comparative Diagnostic Platforms:

1.  .Jiang, F., Jiang, Y., Zhi, H., et al. (2017). Artificial

    > intelligence in healthcare: past, present and future. Stroke and
    > Vascular Neurology, 2(4), 230-243.

2.  Digital Health Platforms and Diagnostic Tools:

**Online Diagnostic Platforms:**

1.  Ada Health. (2024). Symptom Assessment Platform.

    > [[https://ada.com/]{.underline}](https://ada.com/)

2.  Babylon Health. (2024). AI-Powered Healthcare Services.

    > [[https://www.babylonhealth.com/]{.underline}](https://www.babylonhealth.com/)

3.  Your.MD. (2024). Personal Health Companion.
    > [[https://your.md/]{.underline}](https://your.md/)

**Symptom Checking Tools:**

10\. WebMD Symptom Checker. (2024).
[[https://symptoms.webmd.com/]{.underline}](https://symptoms.webmd.com/)

11. Mayo Clinic Symptom Checker. (2024).
    > [[https://www.mayoclinic.org/symptom-checker/select-symptom/itt-20009075]{.underline}](https://www.mayoclinic.org/symptom-checker/select-symptom/itt-20009075)

**Regulatory and Ethical Frameworks:**

1.  World Health Organization. (2021). Ethics and Governance of

    > Artificial Intelligence for Health. WHO Guidance Document.

2.  National Institutes of Health. (2022). Digital Health Technology
    > Evaluation Guidelines.
    > [[https://www.nih.gov/]{.underline}](https://www.nih.gov/)
