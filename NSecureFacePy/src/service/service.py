import os
from flask import Flask, request, render_template
from . import db


def create_app(instance_path):
    app = Flask(__name__, instance_relative_config=True, instance_path=instance_path)

    app.config.from_pyfile('config.py', silent=False)

    db.init_app(app)

    try:
        print(app.instance_path)
        if not os.path.exists(app.instance_path):
            os.makedirs(app.instance_path)
    except OSError as e:
        print(e)

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
            return render_template('error.html', error_message="failed to register - %s" % e)
        finally:
            db.close_db()

        return render_template('status.html', status='successfully registered')

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
            print(e)
        finally:
            db.close_db()

        return render_template('status.html', status='can not find any clients')

    @app.route('/count', methods=['GET'])
    def get_count():
        try:
            client_name = request.args.get('client_name')
            machine_name = request.args.get('machine_name')
            username = request.args.get('username')

            count = db.count(client_name, username, machine_name)

            return {'count': count}
        except Exception as e:
            return render_template('error.html', error_message="failed to get count - %s" % e)
        finally:
            db.close_db()

    @app.route('/delete/<client_name>', methods=['DELETE'])
    def delete_by_client(client_name):
        try:
            if db.remove_by_client(client_name) is True:
                return render_template('status.html', status='{} successfully delete'.format(client_name))
        except Exception as e:
            return render_template('error.html', error_message="failed to delete by client %s - %s" % (client_name, e))

        return render_template('status.html', status='failed to delete {}'.format(client_name))

    @app.route('/delete/<client_name>/<username>/<machine_name>', methods=['DELETE'])
    def delete(client_name, username, machine_name):
        try:
            if db.remove(client_name, username, machine_name) is True:
                return render_template(
                    'status.html', status='[{},{},{}] successfully delete'.format(client_name, username, machine_name)
                )
        except Exception as e:
            return render_template(
                'error.html',
                error_message="failed to delete by client [{},{},{}] - {}".format(client_name, username, machine_name, e)
            )

        return render_template(
            'status.html',
            status='failed to delete [{},{},{}]'.format(client_name, username, machine_name)
        )

    return app
