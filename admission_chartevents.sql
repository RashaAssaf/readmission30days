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
and A.HADM_ID=%s;