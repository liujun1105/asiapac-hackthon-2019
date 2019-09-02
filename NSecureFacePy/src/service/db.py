import sqlite3
import click
from flask import current_app, g
from flask.cli import with_appcontext


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    else:
        print('db in g')
    return g.db


def register(client_name, username, machine_name):
    db = get_db()
    try:
        db.execute("INSERT INTO ClientAuthInfo VALUES ('%s', '%s', '%s')" % (client_name, username, machine_name))
        db.commit()
    except Exception as e:
        db.rollback()
        raise e


def deregister(client_name, username=None, machine_name=None):
    if client_name is None:
        print("client_name is required")
        return

    db = get_db()
    try:
        if username is not None and machine_name is not None:
            db.execute(
                "DELETE FROM ClientAuthInfo WHERE client_name = '%s' AND username = '%s' AND machine_name = '%s'"
                % (client_name, username, machine_name)
            )
        elif username is None:
            db.execute(
                "DELETE FROM ClientAuthInfo WHERE client_name = '%s' AND machine_name = '%s'"
                % (client_name, machine_name)
            )
        elif machine_name is None:
            db.execute(
                "DELETE FROM ClientAuthInfo WHERE client_name = '%s' AND username = '%s'"
                % (client_name, username)
            )
        db.commit()
    except Exception as e:
        db.rollback()
        raise e


def list_all(client_name, username=None, machine_name=None):
    db = get_db()
    try:
        sql = None
        params = ()
        if username is not None and machine_name is not None:
            params = (client_name, username, machine_name,)
            sql = "SELECT * FROM ClientAuthInfo WHERE client_name = ? AND username = ? AND machine_name = ?"
        elif client_name is not None and username is None and machine_name is None:
            params = (client_name,)
            sql = "SELECT * FROM ClientAuthInfo WHERE client_name = ?"
        elif client_name is None and username is None and machine_name is not None:
            params = (client_name, machine_name,)
            sql = "SELECT * FROM ClientAuthInfo WHERE client_name = ? AND machine_name = ?"
        elif client_name is None and machine_name is None and username is not None:
            params = (client_name, username,)
            sql = "SELECT * FROM ClientAuthInfo WHERE client_name = ? AND username = ?"
        else:
            params = ()
            sql = "SELECT * FROM ClientAuthInfo"
        print(sql)

        rows = []
        if len(params) == 0:
            rows = db.execute(sql)
        else:
            rows = db.execute(sql, params)

        results = []
        columns = []
        for row in rows:
            columns = row.keys()
            result_row = {}
            for key in row.keys():
                result_row[key] = row[key]
            results.append(result_row)

        return columns, results

    except Exception as e:
        raise e


def count(client_name, username, machine_name):
    db = get_db()
    try:
        sql = "SELECT count(*) FROM ClientAuthInfo WHERE client_name = ? AND username = ? AND machine_name = ?"
        params = (client_name, username, machine_name,)
        rows = db.execute(sql, params)
        return rows.fetchone()[0]
    except Exception as e:
        raise e


def remove_by_client(client_name):
    """
    remove records from database that matching the given client_name
    """
    db = get_db()
    try:
        pass
        sql = "DELETE FROM ClientAuthInfo WHERE client_name = ?"
        params = (client_name,)
        db.execute(sql, params)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e


def remove(client_name, username, machine_name):
    """
    remove records from database that matching the given client_name, username, machine_name
    """
    db = get_db()
    try:
        pass
        sql = "DELETE FROM ClientAuthInfo WHERE client_name = ? AND username = ? AND machine_name = ?"
        params = (client_name, username, machine_name,)
        db.execute(sql, params)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_app(app):
    app.teardown_appcontext(close_db)


