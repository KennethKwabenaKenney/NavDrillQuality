---
title: "NavDrill Quality: A software tool for spatial QA/QC of underground production drilling"
tags:
  - underground mining
  - drill and blast
  - geospatial analysis
  - QA/QC
  - mining software
authors:
  - name: Kenneth Kwabena Kenney
    affiliation: 1, 2
affiliations:
  - name: Independent Researcher, United States
    index: 1
  - name: Nevada Gold Mines, United States
    index: 2
date: 2026-01-XX
bibliography: paper.bib
---

## Summary

NavDrill Quality (NavDQ) is an open-source software tool developed to support quality assurance and quality control (QA/QC) of underground production drilling and blasting operations. In many underground mining workflows, as-drilled hole data captured by modern drill rigs are recorded in the drill rig’s local coordinate system, making it difficult to integrate these data with mine planning and design environments [@iredes].

NavDQ addresses this limitation by transforming drill navigation data into the mine spatial reference system and generating spatially referenced as-drilled drill hole geometries. The software also preserves drilling metadata captured during the drilling process and organizes these data into analysis-ready outputs suitable for engineering review, visualization, and downstream processing. NavDQ is intended for use by mining engineers, geospatial engineers, and technical personnel involved in drill-and-blast planning and QA/QC workflows.

---

## Statement of Need

Accurate QA/QC of underground production drilling is essential for evaluating drilling accuracy, blast preparation, and post-blast performance [@langefors1978; @jimeno1995]. Traditional methods for acquiring as-drilled hole geometry, such as gyroscopic borehole surveying, are labor-intensive, time-consuming, and often applied inconsistently due to operational constraints [@hustrulid2013]. As a result, many mining operations rely on limited or incomplete as-drilled information when assessing drilling performance.

Recent advances in drill rig technology enable direct measurement of drilling actuals and the capture of associated metadata during the drilling process [@schunnesson2017]. However, these data are commonly recorded in drill-local coordinate systems and remain underutilized without effective spatial transformation and integration into mine planning frameworks. NavDQ was developed to address this gap by enabling systematic spatial integration and QA/QC analysis of drill navigation data using industry-standard formats. The software targets underground mining engineers and researchers who require reproducible, spatially consistent as-drilled data for drilling performance assessment and continuous improvement initiatives.

The development of NavDQ was motivated by practical QA/QC challenges encountered during underground production drilling operations and reconcialiations within an industrial mining environment, including work performed in collaboration with operational engineering teams at Nevada Gold Mines.

---

## State of the Field

Commercial mine planning and drill-and-blast software platforms provide comprehensive tools for drill design, blast modeling, and visualization [@deswik; @datamine]. These systems typically assume that as-drilled hole data are already spatially referenced within the mine coordinate system or rely on external borehole surveying workflows to obtain as-drilled geometry. While effective for planning and visualization, they do not directly address the transformation and QA/QC integration of drill navigation data recorded in drill-local coordinate frames.

NavDQ complements existing mine planning software by focusing specifically on the spatial transformation and QA/QC analysis of drill navigation data. Rather than extending proprietary platforms, NavDQ was developed as a standalone, open-source tool to enable transparent and reproducible processing of drill navigation data using industry-standard formats [@iredes]. This build-rather-than-contribute approach supports independent verification of processing logic and facilitates integration across multiple mine planning environments without reliance on vendor-specific architectures.

---

## Software Design

NavDQ was designed as a modular processing pipeline that separates data ingestion, parsing, spatial transformation, metric preparation, and output generation into distinct stages. This architecture supports traceability between input data and derived outputs while allowing individual components to be extended or adapted for different drilling systems or mine coordinate frameworks.

A key design decision was to treat spatial transformation as a first-class component of the workflow, ensuring that as-drilled hole geometry and associated metadata are preserved consistently during conversion from drill-local coordinates to the mine spatial reference system. The transformation logic incorporates rigid-body alignment methods commonly used in coordinate registration, including approaches based on the Kabsch algorithm [@kabsch1976], ensuring that as-drilled hole geometry and associated metadata are preserved consistently during conversion from drill-local coordinates to the mine spatial reference system [@iliffe2019]. The software prioritizes compatibility with industry-standard data formats, including IREDES, DXF, and CSV, to facilitate integration with existing mine planning, design, and visualization tools. A graphical user interface was included to support practical use by engineering personnel without requiring direct interaction with source code, while maintaining a clear separation between the user interface and core processing logic.

---

## Research Impact Statement

NavDQ provides a reproducible framework for integrating drill navigation data into mine planning coordinate systems and for generating quantitative QA/QC metrics from production drilling data. The software supports both operational engineering workflows and research activities related to drilling performance analysis, blast evaluation, and continuous improvement of drill-and-blast practices [@schunnesson2017].

The open-source release of NavDQ, together with publicly available documentation and example workflows, enables independent evaluation, reuse, and extension by the mining engineering and geospatial research communities. By lowering the barrier to spatial QA/QC analysis of drill navigation data, NavDQ offers credible near-term significance for research studies and operational environments where consistent as-drilled data integration is currently limited.

---

## AI Usage Disclosure

Generative AI tools were used to assist with refining the manuscript text and software documentation. All AI-generated content was reviewed, edited, and verified by the author to ensure technical accuracy, and consistency with the implemented software. No AI tools were used to generate or modify the software’s source code.
