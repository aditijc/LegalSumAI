# LegalSumAI
Generating Robust and Representative Legal Summaries with Large Language Models

## Why Use LegalSumAI?
Legal battles are a frequent and high-risk challenge, with 56\% of US households encountering legal issues annually. The complexity and length of legal documents often make them difficult for the general public to understand. This project aims to generate robust legal summaries from case documents using Large Language Models (LLMs). Our key research question investigates whether we can prompt an LLM to produce accurate, comprehensible summaries of legal cases and opinions. We leverage the Multi-LexSum dataset, which provides expert-authored summaries at three different granularity levels (tiny, short, long). Our proposed two-step pipeline involves generating structured CSV fact sheets using the IRAC method as well as information about the parties involved and other logistical case details, followed by natural language summaries prompted with Chain of Density (CoD) techniques. We evaluate the generated summaries using ROUGE and BERTScore metrics to ensure accuracy and reliability. This approach aims to mitigate LLM hallucinations and improve the accessibility of legal information for the general public.

## Demo
[![Watch the video](https://img.youtube.com/vi/2zS911eINoI/maxresdefault.jpg)](https://youtu.be/2zS911eINoI)
