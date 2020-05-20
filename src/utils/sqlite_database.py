# !C:/Zhuoyi/Study/Github/Chinese_Image_Generation
# -*- coding: utf-8 -*-
# @Time : 5/20/2020 12:16 AM
# @Author : Zhuoyi Huang
# @File : sqlite_database.py

# Definition of database database.db

import sqlite3
import numpy as np
import io
import os
from utils.class_definitions import BoundingBox, CharDef

"""
Converting numpy array to binary and back for insertion in sql
"""


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

DATABASE_PATH = "./database.db"


def open_database():
    # if os.path.exists(DATABASE_PATH):
    #     print("YES!")
    # else:
    #     print("NO")
    conn = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()
    return conn, cur


def close_database(conn):
    conn.commit()
    conn.close()


def create_database():
    print("Creating database")
    (conn, cur) = open_database()
    cur.executescript("""
        CREATE table if not exists Character(
            uid Text primary key,
            lid Integer not null,
            compsLen Integer not null,
            comp1 Text not null,
            comp2 Text,
            comp3 Text,
            comp4 Text,
            comp5 Text,
            pdef Integer, 
            Foreign key (pdef) references PreciseDef(pdefid)
        );

        CREATE table if not exists PreciseDef(
            pdefid Integer primary key,
            boxId1 Integer not null,
            boxId2 Integer not null,
            boxId3 Integer,
            boxId4 Integer,
            boxId5 Integer,
            Foreign key (boxId1) references Box(boxId),
            Foreign key (boxId2) references Box(boxId),
            Foreign key (boxId3) references Box(boxId),
            Foreign key (boxId4) references Box(boxId),
            Foreign key (boxId5) references Box(boxId)
        );

        CREATE table if not exists Box(
            boxId Integer primary key,
            x Integer not null,
            y Integer not null,
            dx Integer not null,
            dy Integer not null
        );
        """)
    close_database(conn)
    print("Done")


def add_precise_def(char_def):
    (conn, cur) = open_database()
    box_ids = []
    for i in range(5):
        if i < char_def.compsLen:
            box = char_def.boxes[i]
            cur.execute("Insert Into Box(x,y,dx,dy) values (?,?,?,?)", (box.x, box.y, box.dx, box.dy))
            box_ids.append(cur.lastrowid)
        else:
            box_ids.append(None)
    cur.execute("Insert Into PreciseDef(boxId1, boxId2, boxId3, boxId4, boxId5) values (?,?,?,?,?)", box_ids)
    p_def_id = cur.lastrowid
    cur.execute("Update Character set pdef = ? where uid = ?", (p_def_id, char_def.uid))
    close_database(conn)


def insert_precise_def(char_def, cur):
    if char_def.preciseDef:
        box_ids = []
        for i in range(5):
            if i < char_def.compsLen:
                box = char_def.boxes[i]
                cur.execute("Insert Into Box(x,y,dx,dy) values (?,?,?,?)", (box.x, box.y, box.dx, box.dy))
                box_ids.append(cur.lastrowid)
            else:
                box_ids.append(None)
        cur.execute("Insert Into PreciseDef(boxId1, boxId2, boxId3, boxId4, boxId5) values (?,?,?,?,?)", box_ids)
        return cur.lastrowid
    else:
        return None


def insert_char(char_def):
    (conn, cur) = open_database()
    p_def_id = insert_precise_def(char_def, cur)
    char = [char_def.uid, char_def.lid, char_def.compsLen]
    for i in range(5):
        if i < char_def.compsLen:
            char.append(char_def.compIds[i])
        else:
            char.append(None)
    char.append(p_def_id)
    cur.execute("Insert Into Character(uid,lid,compsLen, comp1, comp2, comp3, comp4, comp5, pdef) values (?,?,?,?,?,?,?,?,?)", char)
    close_database(conn)


def update_char(char_def):
    (conn, cur) = open_database()
    char = [char_def.lid, char_def.compsLen]
    for i in range(5):
        if i < char_def.compsLen:
            char.append(char_def.compIds[i])
        else:
            char.append(None)
    char.append(char_def.uid)
    cur.execute("Update Character set lid = ?, compsLen = ?, comp1 = ?, comp2 = ?, comp3 = ?, comp4 = ?, comp5 = ? where uid = ?", char)
    close_database(conn)


def get_unicode_list():
    (conn, cur) = open_database()
    unicodes = cur.execute("Select uid from Character").fetchall()
    close_database(conn)
    return [uid[0] for uid in unicodes]


def get_char_def(uid):
    (conn, cur) = open_database()
    char_def_as_list = cur.execute("Select lid, compsLen, comp1, comp2, comp3, comp4, comp5, pdef from Character where uid = ?", (uid,)).fetchone()
    lid = char_def_as_list[0]
    comps_length = char_def_as_list[1]
    comp_ids = []
    for i in range(comps_length):
        comp_ids.append(char_def_as_list[2+i])
    precise_def = char_def_as_list[7] is not None
    if precise_def:
        p_def = cur.execute("Select boxId1, boxId2, boxId3, boxId4, boxId5 from PreciseDef where pdefid = ?", (char_def_as_list[7],)).fetchone()
        boxes = []
        for i in range(comps_length):
            (x, y, dx, dy) = cur.execute("Select x, y, dx, dy from Box where boxId = ?", (p_def[i],)).fetchone()
            boxes.append(BoundingBox(x, y, dx, dy))
        char_def = CharDef(uid, lid, comps_length, comp_ids, True, boxes)
    else:
        char_def = CharDef(uid, lid, comps_length, comp_ids)
    close_database(conn)
    return char_def


def get_char_def_dic():
    char_def_dic = {}
    b_count = 0
    rd_count = 0
    pd_count = 0
    (conn, cur) = open_database()
    chars = cur.execute("Select uid, lid, compsLen, comp1, comp2, comp3, comp4, comp5, pdef from Character").fetchall()
    for char in chars:
        uid = char[0]
        lid = char[1]
        comps_length = char[2]
        comp_ids = []
        for i in range(comps_length):
            comp_ids.append(char[3+i])
        precise_def = char[8] is not None
        if precise_def:
            p_def = cur.execute("Select boxId1, boxId2, boxId3, boxId4, boxId5 from PreciseDef where pdefid = ?",
                                (char[8],)).fetchone()
            boxes = []
            for i in range(comps_length):
                (x, y, dx, dy) = cur.execute("Select x, y, dx, dy from Box where boxId = ?", (p_def[i],)).fetchone()
                boxes.append(BoundingBox(x, y, dx, dy))
            pd_count += 1
            char_def = CharDef(uid, lid, comps_length, comp_ids, True, boxes)
        else:
            if lid == 0:
                b_count += 1
            else:
                rd_count += 1
            char_def = CharDef(uid, lid, comps_length, comp_ids)
        char_def_dic.update({uid: char_def})
    close_database(conn)
    # print("Amount of base characters: ", bCount)
    # print("Amount of roughly defined characters: ", rdCount)
    # print("Amount of precisely defined characters: ", pdCount)
    return char_def_dic


def main():
    create_database()


if __name__ == '__main__':
    main()
