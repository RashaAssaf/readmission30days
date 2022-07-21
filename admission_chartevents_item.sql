select
    min(case when ITEMID in (%s) and VALUENUM > 0 then VALUENUM else null end),
    max(case when ITEMID in (%s) and VALUENUM > 0 then VALUENUM else null end),
    avg(case when ITEMID in (%s) and VALUENUM > 0 then VALUENUM else null end)
from CHARTEVENTS E inner join ADMISSIONS A on E.HADM_ID = A.HADM_ID
where itemid in (%s)
and A.HADM_ID=%s;