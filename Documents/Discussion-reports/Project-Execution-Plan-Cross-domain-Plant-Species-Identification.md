### Project Execution Plan: Cross-domain Plant Species Identification

**Date:** 10th of November, 2025

---

### 1. Executive Summary

This document outlines a comprehensive project execution plan for the COS30082 Applied Machine Learning project, "Cross-domain Plant Species Identification." The primary objective is to develop and evaluate deep learning models for identifying plant species from field images, using herbarium specimens as the primary training data. This plan details a structured approach for a five-member team, focusing on a balanced and equitable distribution of workload. The strategy involves assigning each member a core development task (a machine learning model or the user interface) and a key deliverable task (report sections or presentation management). The project is divided into four distinct phases—Foundation, Core Development, Integration, and Finalization—to ensure a clear workflow, foster collaboration, and meet all submission requirements by the deadline of 28th November 2025.

---

### 2. Introduction

#### 2.1. Project Overview
The project requires the implementation and evaluation of three distinct approaches to solve a cross-domain classification problem: two baseline models (a Mix-stream CNN and a DINOv2 feature extractor) and one novel deep learning solution. The final output includes the trained models integrated into a user-friendly interface, a comprehensive project report, and a video presentation.

#### 2.2. Guiding Principles
The core principle of this plan is to ensure equal contribution from all five group members. This is achieved by:
*   Assigning every member a significant hands-on development role.
*   Distributing responsibility for the final deliverables (report, presentation, UI) across the team.
*   Creating interdependencies that necessitate continuous collaboration and communication.

---

### 3. Role Allocation and Task Distribution

To ensure a balanced workload, each team member is assigned a primary development task and a secondary deliverable task.

| Member | Core Development Task | Deliverable & Reporting Task |
| :--- | :--- | :--- |
| **Member 1** | **New Approach (Lead Researcher)** | - Writes Methodology for New Approach.<br>- Leads creation of the Presentation slides. |
| **Member 2** | **New Approach (Lead Implementer)** | - Writes Results/Discussion for New Approach.<br>- Manages Git repository and code integration. |
| **Member 3** | **Baseline 1: Mix-stream CNN** | - Writes Introduction & Methodology for Baseline 1.<br>- Gathers all results into final tables/charts. |
| **Member 4** | **Baseline 2: DINOv2 Feature Extractor** | - Writes Methodology & Results for Baseline 2.<br>- Records and edits the final presentation video. |
| **Member 5** | **User Interface (UI) Development** | - Writes the UI section of the report.<br>- Compiles and formats the final PDF report. |

---

### 4. Project Execution Plan: A Phased Approach

The project timeline is structured into four distinct phases to manage progress effectively.

#### 4.1. Phase 1: Foundation & Setup (Weeks 1-3)
This initial phase is a collaborative effort involving all team members to build a strong foundation.
*   **Project Management:** Member 2 will initialize the Git repository and grant access. The team will establish a primary communication channel (e.g., Discord) and a task board (e.g., Trello).
*   **Dataset Analysis:** All members will independently analyze the dataset, followed by a group meeting to discuss findings on class distribution, image characteristics, and the herbarium-field pair challenge.
*   **Shared Codebase Development:** The entire team will contribute to creating a universal data loading and image preprocessing pipeline to ensure consistency across all models.
*   **Literature Review:** All members will research existing methods for cross-domain adaptation and class imbalance to support the design of the new approach.

#### 4.2. Phase 2: Core Development (Weeks 4-8)
During this phase, members will work in parallel on their assigned core development tasks.
*   **New Approach Team (Member 1 & 2):** Will collaboratively design, code, and begin training the novel deep learning model. Member 1 will focus on the research and experimental design, while Member 2 will lead the implementation.
*   **Baseline 1 (Member 3):** Will implement, train, and evaluate the Mix-stream CNN model.
*   **Baseline 2 (Member 4):** Will implement the DINOv2 feature extraction pipeline and train various traditional machine learning classifiers on the generated embeddings.
*   **UI Development (Member 5):** Will design UI mock-ups, develop the graphical interface, and prepare the back-end for model integration.

#### 4.3. Phase 3: Integration and Deliverables (Weeks 9-12)
The focus shifts from pure development to integrating components and producing the final deliverables.
*   **Model Integration:** Members 1, 3, and 4 will finalize their models and hand them over to Member 2, who will work with Member 5 to integrate them into the UI.
*   **Report Writing:** Each member will write their assigned sections concurrently. Member 5 will collate these sections into the final report document and ensure consistent formatting. Member 3 will be responsible for standardizing all result tables and charts.
*   **Presentation Preparation:** Member 1 will create the master presentation slides. All members will prepare their respective sections. Member 4 will manage the recording and editing of the 8-minute video presentation.

---

### 5. Contribution and Workload Management

This plan is explicitly designed to prevent workload imbalance. By assigning every member a substantial coding task and a critical role in the final report and presentation, we ensure that no individual is idle at any stage of the project. Regular weekly sync-up meetings will be held to monitor progress, address roadblocks, and maintain alignment. The Git repository, managed by Member 2, will serve as a transparent record of each member's coding contributions.

---

### 6. Conclusion

This project execution plan provides a clear and structured roadmap for the successful completion of the "Cross-domain Plant Species Identification" project. It establishes well-defined roles, distributes tasks equitably, and promotes a collaborative environment. By adhering to this phased approach, the team is well-positioned to meet all project requirements, effectively manage the workload, and deliver a high-quality project on time.