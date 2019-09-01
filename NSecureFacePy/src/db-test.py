import sqlite3


class SecureServiceDB:

    def __init__(self):
        self.__conn = None

    def init_db(self, db_file):
        self.__conn = None
        try:
            self.__conn = sqlite3.connect(db_file)
            print("SQLite Version => %s" % sqlite3.version)
        except Exception as e:
            print(e)

    def create_tables(self):
        if self.__conn:
            sql = "CREATE TABLE ClientAuthInfo (client_name TEXT, username TEXT, machine_name TEXT, PRIMARY KEY (client_name, username, machine_name))"
            try:
                print("executing [%s]" % sql)
                self.__conn.execute(sql)
                self.__conn.commit()
            except Exception as e:
                print(e)
                self.__conn.rollback()

    def drop_tables(self):
        if self.__conn:
            sql = "DROP TABLE IF EXISTS ClientAuthInfo"
            try:
                print("executing [%s]" % sql)
                self.__conn.execute(sql)
                self.__conn.commit()
            except Exception as e:
                print(e)
                self.__conn.rollback()

    def register(self, client_name, username, machine_name):
        if self.__conn:
            try:
                sql = "INSERT INTO ClientAuthInfo VALUES ('%s', '%s', '%s')" % (client_name, username, machine_name)
                print("executing [%s]" % sql)
                self.__conn.execute(sql)
                self.__conn.commit()
            except sqlite3.IntegrityError as e:
                print(e)
                self.__conn.rollback()
                "failed"

            return "success"

    def search_by_client(self, client_name):
        if self.__conn:
            sql = "SELECT * FROM ClientAuthInfo WHERE client_name = :client_name"

            try:
                print("executing [%s]" % sql)
                rows = self.__conn.execute(sql, {"client_name": client_name})

                for row in rows:
                    print(row)
            except Exception as e:
                print(e)

        return ""

    def shutdown(self):
        if self.__conn:
            self.__conn.close()


if __name__ == '__main__':
    ssdb = SecureServiceDB()
    ssdb.init_db('face-auth.db')
    # ssdb.drop_tables()
    # ssdb.create_tables()
    ssdb.register('camera-app', 'liujunju', 'shawd730058')
    ssdb.search_by_client('camera-app')
    ssdb.shutdown()
