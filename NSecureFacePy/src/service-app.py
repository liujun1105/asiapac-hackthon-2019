import service
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Service App')
    parser.add_argument('--instance', help='service instance directory', type=str, required=True)

    args = parser.parse_args()

    app = service.create_app(args.instance)
    app.run('0.0.0.0')
