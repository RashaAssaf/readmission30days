-- Get diagnosis codes (ICD)
select A.SUBJECT_ID, A.HADM_ID, A.ADMITTIME, A.DISCHTIME, DI.ICD9_CODE, DID.SHORT_TITLE, DID.LONG_TITLE from ADMISSIONS A
inner join DIAGNOSES_ICD DI on A.HADM_ID = DI.HADM_ID
inner join D_ICD_DIAGNOSES DID on DI.ICD9_CODE = DID.ICD9_CODE;

-- Get Admissions
select A.SUBJECT_ID, A.HADM_ID, A.ADMITTIME, A.DISCHTIME from ADMISSIONS A
    inner join PATIENTS P on A.SUBJECT_ID = P.SUBJECT_ID
where A.ADMISSION_TYPE<>'NEWBORN'
order by P.SUBJECT_ID;

-- Get Diagnosis Related Groups (DRG)
select A.SUBJECT_ID, A.HADM_ID, A.ADMITTIME, A.DISCHTIME, D.DRG_CODE, D.DESCRIPTION from ADMISSIONS A
    inner join PATIENTS P on A.SUBJECT_ID = P.SUBJECT_ID
    inner join DRGCODES D on A.HADM_ID = D.HADM_ID
where A.ADMISSION_TYPE<>'NEWBORN';

-- Get labs
select A.SUBJECT_ID, A.HADM_ID, DL.CATEGORY, DL.LABEL, DL.FLUID, L.VALUE, L.VALUENUM from ADMISSIONS A
inner join LABEVENTS L on A.HADM_ID = L.HADM_ID
inner join D_LABITEMS DL on L.ITEMID = DL.ITEMID;

-- Get medications
select A.SUBJECT_ID, A.HADM_ID, P.DRUG, P.PROD_STRENGTH, P.ROUTE from ADMISSIONS A
inner join PRESCRIPTIONS P on A.HADM_ID = P.HADM_ID;

-- Get procedures
select A.SUBJECT_ID, A.HADM_ID, PI.ICD9_CODE, PI.SEQ_NUM, DIP.SHORT_TITLE, DIP.LONG_TITLE from ADMISSIONS A
inner join PROCEDURES_ICD PI on PI.HADM_ID=A.HADM_ID
inner join D_ICD_PROCEDURES DIP on PI.ICD9_CODE = DIP.ICD9_CODE;

-- Get relevant chart items
select * from D_ITEMS where itemid in (615,618,220210,224690, -- RespRate
                                      807,811,1529,3745,3744,225664,220621,226537, -- Glucose
                                      211,220045, -- HR
                                      51,442,455,6701,220179,220050, -- SysBP
                                      8368,8440,8441,8555,220180,220051, -- DiasBP
                                      223761,678,223762,676, -- Temp
                                      763 -- Daily Weight
                                      );

-- Aggregate chart events by visit (compute min, max and mean)
select
       min(case when ITEMID in (615,618,220210,224690) and VALUENUM > 0 then VALUENUM else null end) as MinRespRate,
       max(case when ITEMID in (615,618,220210,224690) and VALUENUM > 0 then VALUENUM else null end) as MaxRespRate,
       avg(case when ITEMID in (615,618,220210,224690) and VALUENUM > 0 then VALUENUM else null end) as MeanRespRate,
       min(case when ITEMID in (807,811,1529,3745,3744,225664,220621,226537) and VALUENUM > 0 then VALUENUM else null end) as MinGlucose,
       max(case when ITEMID in (807,811,1529,3745,3744,225664,220621,226537) and VALUENUM > 0 then VALUENUM else null end) as MaxGlucose,
       avg(case when ITEMID in (807,811,1529,3745,3744,225664,220621,226537) and VALUENUM > 0 then VALUENUM else null end) as MeanGlucose,
       min(case when ITEMID in (211,220045) and VALUENUM > 0 then VALUENUM else null end) as MinHeartRate,
       max(case when ITEMID in (211,220045) and VALUENUM > 0 then VALUENUM else null end) as MaxHeartRate,
       avg(case when ITEMID in (211,220045) and VALUENUM > 0 then VALUENUM else null end) as MeanHeartRate,
       min(case when ITEMID in (51,442,455,6701,220179,220050) and VALUENUM > 0 then VALUENUM else null end) as MinSysBP,
       max(case when ITEMID in (51,442,455,6701,220179,220050) and VALUENUM > 0 then VALUENUM else null end) as MaxSysBP,
       avg(case when ITEMID in (51,442,455,6701,220179,220050) and VALUENUM > 0 then VALUENUM else null end) as MeanSysBP,
       min(case when ITEMID in (8368,8440,8441,8555,220180,220051) and VALUENUM > 0 then VALUENUM else null end) as MinDiasBP,
       max(case when ITEMID in (8368,8440,8441,8555,220180,220051) and VALUENUM > 0 then VALUENUM else null end) as MaxDiasBP,
       avg(case when ITEMID in (8368,8440,8441,8555,220180,220051) and VALUENUM > 0 then VALUENUM else null end) as MeanDiasBP,
       min(case when ITEMID in (223761,678,223762,676) and VALUENUM > 0 then VALUENUM else null end) as MinTemp,
       max(case when ITEMID in (223761,678,223762,676) and VALUENUM > 0 then VALUENUM else null end) as MaxTemp,
       avg(case when ITEMID in (223761,678,223762,676) and VALUENUM > 0 then VALUENUM else null end) as MeanTemp,
       min(case when ITEMID in (763) and VALUENUM > 0 then VALUENUM else null end) as MinWeight,
       max(case when ITEMID in (763) and VALUENUM > 0 then VALUENUM else null end) as MaxWeight,
       avg(case when ITEMID in (763) and VALUENUM > 0 then VALUENUM else null end) as MeanWeight
from CHARTEVENTS E inner join ADMISSIONS A on E.HADM_ID = A.HADM_ID
where itemid in (
            615,618,220210,224690, -- RespRate
            807,811,1529,3745,3744,225664,220621,226537, -- Glucose
            211,220045, -- HeartRate
            51,442,455,6701,220179,220050, -- SysBP
            8368,8440,8441,8555,220180,220051, -- DiasBP
            223761,678,223762,676, -- Temp
            763 -- Daily Weight
        )
and A.HADM_ID=111970;

select count(*) from DIAGNOSES_ICD;