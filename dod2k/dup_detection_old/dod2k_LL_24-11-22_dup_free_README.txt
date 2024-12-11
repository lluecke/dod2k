 Decisions for duplicate candidate pairs. 
 Modified by Lucie Luecke (LL)
 E-Mail: lluecke@ed.ac.uk
 Modified on: 2024-11-22 13:39:06.663428 (UTC)
 Test run to check if hierarchy implementation went well and code runs fine otherwise (with starting/stopping). No bugs. However selection process for automated choices was modified throughout running of the code to relax lat/lon matching and at duplicate 58 included a check for length of data into the perfect correlation criterion. This means that in some cases records were automatically chosen by the algorithm even though the time coordinate did not match in length for the two records in question. 
