rm(list = ls(all = TRUE))
source("fn.base.R")
gc()

##############################################################################################
fn.load.data("sub")
fn.load.data("pred.xg.1")
fn.load.data("pred.xg.2")
fn.load.data("pred.xg.3")
fn.load.data("pred.xg.4")
fn.load.data("pred.xg.5")
fn.load.data("pred.xg.6")
fn.load.data("pred.xg.7")
fn.load.data("pred.xg.8")
fn.load.data("pred.xg.9")

fn.load.data("pred.xg.1D")
fn.load.data("pred.xg.2D")
fn.load.data("pred.xg.3D")
fn.load.data("pred.xg.4D")
fn.load.data("pred.xg.5D")
fn.load.data("pred.xg.6D")
fn.load.data("pred.xg.7D")
fn.load.data("pred.xg.8D")
fn.load.data("pred.xg.9D")

pred.test.all  <- matrix( 0 ,nrow=nrow(sub)  , ncol=104 )
pred.test.all[,1:37]  <- 0.35*pred.xg.1D$test+0.65*pred.xg.1$test
pred.test.all[,38:48] <- 0.35*pred.xg.2D$test+0.65*pred.xg.2$test
pred.test.all[,49:51] <- 0.35*pred.xg.3D$test+0.65*pred.xg.3$test
pred.test.all[,52:76] <- 0.35*pred.xg.4D$test+0.65*pred.xg.4$test
pred.test.all[,77:79] <- 0.35*pred.xg.5D$test+0.65*pred.xg.5$test
pred.test.all[,80:82] <- 0.35*pred.xg.6D$test+0.65*pred.xg.6$test
pred.test.all[,83:87] <- 0.35*pred.xg.7D$test+0.65*pred.xg.7$test
pred.test.all[,88:96] <- 0.35*pred.xg.8D$test+0.65*pred.xg.8$test
pred.test.all[,97:104]<- 0.35*pred.xg.9D$test+0.65*pred.xg.9$test
gc()

sum(pred.test.all==0)
sub[ , 2:105  ] <- pred.test.all
sub[ sub==0 ] <- 1e-16
sub$X[47544] <- 0

write.csv( sub , 'submission/ens1.csv' , quote=FALSE,row.names=FALSE  )

#Now replace ens1.csv header with original SubmissionFormat.csv header
#That's mainly because R don't accept special characters in headers

# Original Header
#,Function__Aides Compensation,Function__Career & Academic Counseling,Function__Communications,Function__Curriculum Development,Function__Data Processing & Information Services,Function__Development & Fundraising,Function__Enrichment,Function__Extended Time & Tutoring,Function__Facilities & Maintenance,Function__Facilities Planning,"Function__Finance, Budget, Purchasing & Distribution",Function__Food Services,Function__Governance,Function__Human Resources,Function__Instructional Materials & Supplies,Function__Insurance,Function__Legal,Function__Library & Media,Function__NO_LABEL,Function__Other Compensation,Function__Other Non-Compensation,Function__Parent & Community Relations,Function__Physical Health & Services,Function__Professional Development,Function__Recruitment,Function__Research & Accountability,Function__School Administration,Function__School Supervision,Function__Security & Safety,Function__Social & Emotional,Function__Special Population Program Management & Support,Function__Student Assignment,Function__Student Transportation,Function__Substitute Compensation,Function__Teacher Compensation,Function__Untracked Budget Set-Aside,Function__Utilities,Object_Type__Base Salary/Compensation,Object_Type__Benefits,Object_Type__Contracted Services,Object_Type__Equipment & Equipment Lease,Object_Type__NO_LABEL,Object_Type__Other Compensation/Stipend,Object_Type__Other Non-Compensation,Object_Type__Rent/Utilities,Object_Type__Substitute Compensation,Object_Type__Supplies/Materials,Object_Type__Travel & Conferences,Operating_Status__Non-Operating,"Operating_Status__Operating, Not PreK-12",Operating_Status__PreK-12 Operating,Position_Type__(Exec) Director,Position_Type__Area Officers,Position_Type__Club Advisor/Coach,Position_Type__Coordinator/Manager,Position_Type__Custodian,Position_Type__Guidance Counselor,Position_Type__Instructional Coach,Position_Type__Librarian,Position_Type__NO_LABEL,Position_Type__Non-Position,Position_Type__Nurse,Position_Type__Nurse Aide,Position_Type__Occupational Therapist,Position_Type__Other,Position_Type__Physical Therapist,Position_Type__Principal,Position_Type__Psychologist,Position_Type__School Monitor/Security,Position_Type__Sec/Clerk/Other Admin,Position_Type__Social Worker,Position_Type__Speech Therapist,Position_Type__Substitute,Position_Type__TA,Position_Type__Teacher,Position_Type__Vice Principal,Pre_K__NO_LABEL,Pre_K__Non PreK,Pre_K__PreK,Reporting__NO_LABEL,Reporting__Non-School,Reporting__School,Sharing__Leadership & Management,Sharing__NO_LABEL,Sharing__School Reported,Sharing__School on Central Budgets,Sharing__Shared Services,Student_Type__Alternative,Student_Type__At Risk,Student_Type__ELL,Student_Type__Gifted,Student_Type__NO_LABEL,Student_Type__Poverty,Student_Type__PreK,Student_Type__Special Education,Student_Type__Unspecified,Use__Business Services,Use__ISPD,Use__Instruction,Use__Leadership,Use__NO_LABEL,Use__O&M,Use__Pupil Services & Enrichment,Use__Untracked Budget Set-Aside

##############################################################################################




