import os
import sys
import multiprocessing as mp
import numpy as np
import functools
from random import sample
import utils
import config


def load_icd9_embeddings():
    filename = os.path.join("..", "claims_codes_hs_300.txt")
    embeddings = list()
    codes = list()

    with open(filename, 'rb') as fh:
        next(fh)

        for line in fh:
            fields = line.split()
            codes.append(fields[0].decode('utf-8').replace("IDX_", ""))
            embeddings.append(np.array([float(x) for x in fields[1:]]))

    return codes, np.asarray(embeddings)


def format_icd9_code(icd9):
    if icd9 is None:
        return None

    if '.' not in icd9:
        if icd9.startswith('E') and len(icd9) > 4:
            icd9 = icd9[:4] + '.' + icd9[4:]
        elif len(icd9) > 3:
            icd9 = icd9[:3] + '.' + icd9[3:]

    return icd9


def process_admissions():
    conn = utils.get_connection()
    db = conn.cursor()
    admissions = {}
    adm_query = utils.get_query("patient_admissions")
    db.execute(adm_query)
    results = db.fetchall()

    for i in range(1, len(results)):
        subject_id, hadm_id, death_time, admit_time, disch_time, admit_loc, disch_loc = results[i]
        prev_subject_id, prev_hadm_id, prev_death_time, prev_admit_time, prev_disch_time, prev_admit_loc, prev_disch_loc = results[i-1]
        next_admit_dt = None
        death_dt = None

        if subject_id == prev_subject_id:
            next_admit_dt = (admit_time - prev_disch_time).days

            if disch_loc == "DEAD/EXPIRED":
                death_dt = (death_time - prev_disch_time).days

        admissions[prev_hadm_id] = {
            "subject_id": prev_subject_id,
            "hadm_id": prev_hadm_id,
            "death_dt": death_dt,
            "admit_time": prev_admit_time,
            "disch_time": prev_disch_time,
            "admit_loc": prev_admit_loc,
            "disch_loc": prev_disch_loc,
            "next_admit_dt": next_admit_dt,
            "next_admit_loc": None if not next_admit_dt else admit_loc
        }

    print("Processed {} admissions".format(len(admissions)))
    return admissions


def link_icdcodes(admissions):
    query = "select ICD9_CODE from DIAGNOSES_ICD where HADM_ID=%s"
    conn = utils.get_connection()
    db = conn.cursor()

    for hadm_id, data in admissions.items():
        db.execute(query, (hadm_id, ))
        drg_codes = db.fetchall()
        data['icd9_codes'] = [drg_code[0] for drg_code in drg_codes]
        data['formatted_icd9_codes'] = [format_icd9_code(drg_code[0]) for drg_code in drg_codes]

    return admissions


def link_gender(admissions):
    query = "select GENDER from PATIENTS where SUBJECT_ID=%s"
    conn = utils.get_connection()
    db = conn.cursor()

    for data in admissions.values():
        db.execute(query, (data['subject_id'], ))
        results = db.fetchone()
        data["gender"] = 0 if results[0].strip() == "M" else 1

    return admissions


def link_chartevents(admissions):
    event_query = utils.get_query("admission_chartevents_item")
    conn = utils.get_connection()
    db = conn.cursor()
    num_results = len(admissions)

    for i, (hadm_id, data) in enumerate(admissions.items()):
        print('Processing HADM_ID {} -- {}/{}'.format(hadm_id, i, num_results))
        data["chart_events"] = list()

        for event, itemids in config.itemids.items():
            itemids_str = ','.join([str(itemid) for itemid in itemids])
            db.execute(event_query, (itemids_str, itemids_str, itemids_str, itemids_str, hadm_id))
            event_results = db.fetchone()
            data["chart_events"] += [0 if not val else float(val) for val in list(event_results)]

    return admissions


def aggregate_embeddings(admission, codes, embeddings):
    indices = list()

    for code in admission['formatted_icd9_codes']:
        try:
            indices.append(codes.index(code))
        except:
            print("OOV: embeddings for code {} is not found".format(code))
            sys.stdout.flush()

    return embeddings[indices,:].sum(0)


def transform(admissions, codes, embeddings):
    pool = mp.Pool(processes=mp.cpu_count())
    icd9_matrix = pool.map(functools.partial(
        aggregate_embeddings,
        codes=codes,
        embeddings=embeddings), admissions.values())
    chartevent_matrix = np.asarray([a['chart_events'] for a in admissions.values()])
    gender_matrix = np.asarray([a['gender'] for a in admissions.values()]).reshape(-1, 1)
    data_matrix = np.hstack((icd9_matrix, gender_matrix, chartevent_matrix))

    return np.asarray(data_matrix)


def split_admissions(admissions):
    #Get list of admissions with next reamdission is within 30 days
    postives = {k: v for k, v in admissions.items() if v['next_admit_dt'] and v['next_admit_dt'] <= 30}

    #Get list of admissions where patient got discharged and died within
    postives.update({k: v for k, v in admissions.items() if v['death_dt'] and v['death_dt'] <= 30})

    #Sample negative admissions
    negative_hadm_ids = [k for k, v in admissions.items() if
                         (not v['next_admit_dt'] or v['next_admit_dt'] > 30)]
    sampled_negative_hadm_id = sample(negative_hadm_ids, len(postives.keys()))
    negatives = {hadm_id: admissions[hadm_id] for hadm_id in sampled_negative_hadm_id}

    return postives, negatives


if __name__ == "__main__":
    conn = utils.get_connection()
    codes, embeddings = load_icd9_embeddings()
    admissions = process_admissions()
    admissions = link_icdcodes(admissions)
    pos, neg = split_admissions(admissions)

    # Linking gender
    pos = link_gender(pos)
    neg = link_gender(neg)

    # Linking Chart Events to each admission
    pos = link_chartevents(pos)
    neg = link_chartevents(neg)

    conn.close()

    print('Convert data to feature vectors')
    pos_matrix = transform(pos, codes, embeddings)
    pos_matrix = np.hstack((pos_matrix, np.ones((pos_matrix.shape[0], 1))))

    neg_matrix = transform(neg, codes, embeddings)
    neg_matrix = np.hstack((neg_matrix, np.zeros((neg_matrix.shape[0], 1))))

    np.savetxt("../data/pos_admissions_v3.csv", pos_matrix, delimiter=",")
    np.savetxt("../data/neg_admissions_v3.csv", neg_matrix, delimiter=",")

