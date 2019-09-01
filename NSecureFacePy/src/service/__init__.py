import os

from flask import Flask, request, render_template


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_pyfile('config.py', silent=False)

    try:
        print(app.instance_path)
        if not os.path.exists(app.instance_path):
            os.makedirs(app.instance_path)
    except OSError as e:
        print(e)

    from . import db
    db.init_app(app)

    @app.route('/status')
    def status():
        return "connected"

    @app.route('/register', methods=['POST'])
    def register():
        try:
            json_data = request.get_json()
            print(json_data)
            client_name = json_data['client_name']
            machine_name = json_data['machine_name']
            username = json_data['username']

            db.register(client_name, username, machine_name)

        except Exception as e:
            return "failed to register - %s" % e
        finally:
            db.close_db()

        return "success"

    @app.route('/clients', methods=['GET'])
    def clients():
        try:
            client_name = request.args.get('client_name')
            machine_name = request.args.get('machine_name')
            username = request.args.get('username')

            print('client_name=%s, machine_name=%s, username=%s' % (client_name, machine_name, username))

            columns, clients = db.list_all(client_name, username, machine_name)
            if len(clients) > 0:
                print(columns)
                return render_template('client-info.html', clients={'columns':columns, 'clients':clients})

        except Exception as e:
            return e
        finally:
            db.close_db()

        return "failed to get client info"

    return app
