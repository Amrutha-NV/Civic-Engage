\# Person 3 — Current Market Evidence



\## Purpose



This folder contains the current market evidence required for the procurement benchmark work in the Civic-Engage project.



The objective is to collect and organize recent, publicly available market-price evidence that can be used to validate and support procurement benchmark prices.



\## Scope



Person 3 is responsible for:



\* Collecting current market-price evidence.

\* Recording the source of each price.

\* Recording the product/item name and relevant specifications.

\* Recording the observed price and date of collection.

\* Keeping the evidence traceable to its original source.

\* Preparing clean data that can later be used to validate the benchmark-price output.



\## Important Rule



The existing ML benchmark system is \*\*frozen\*\*.



Do not modify, retrain, replace, or restructure the existing ML model or its output.



Person 3 work is focused on \*\*external/current market evidence and validation\*\*.



\## Evidence Requirements



Each collected price should, wherever possible, contain:



\* Item/product name

\* Brand or specification, if applicable

\* Unit of measurement

\* Price

\* Source/platform

\* Source URL

\* Date collected

\* Location/market, if available

\* Notes about the listing or quotation



\## Suggested Evidence Sources



Potential sources include:



\* Government procurement portals

\* Government rate/price documents

\* Official supplier websites

\* Established online marketplaces

\* Local supplier/retailer listings

\* Public quotations or catalogues

\* Other reliable and traceable sources



Sources should be selected based on relevance and reliability rather than simply choosing the lowest available price.



\## Data Quality



Before adding an evidence record, check:



1\. The item matches the required procurement item.

2\. The unit is clearly identified.

3\. The price is clearly visible or stated.

4\. The source can be traced.

5\. The information is reasonably current.

6\. Any important assumptions or differences are recorded in the notes.



Do not invent missing prices, specifications, dates, or sources.



\## Folder Structure



The folder will gradually contain:



```text

current\_market\_evidence/

│

├── README.md

├── market\_evidence.csv

├── sources/

└── notes/

```



The exact structure may be expanded later if required.



\## Relationship With the ML Work



The ML benchmark component has already been developed separately.



Person 3 provides independent current-market evidence that can be used to:



\* Compare against benchmark predictions.

\* Identify unusually high or low benchmark values.

\* Check whether predicted price ranges are broadly consistent with current market observations.

\* Document supporting evidence for procurement decisions.



This evidence should not be used to alter the frozen ML implementation unless the project team explicitly decides to do so later.



\## Final Objective



The final output of Person 3 should provide a clean and traceable evidence base showing:



\*\*Procurement Item → Market Evidence → Source → Observed Price → Date → Supporting Notes\*\*



This allows the team to validate the procurement benchmark results using independently collected current-market information.



