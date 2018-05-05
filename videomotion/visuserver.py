import threading
import os
import urllib.parse   # "urllib.parse" is missing in python 2.7
from http.server import BaseHTTPRequestHandler, HTTPServer  # "http.server" is missing in python 2.7
from streams import InputStream
import sys
import socket
from utils import out
import numpy as np


class VisualizationServer:

    def __init__(self, port=8080, html_root=os.getcwd() + os.sep + "web", data_root=os.getcwd()):
        handler = make_handler_class(html_root, data_root)
        self.server = HTTPServer(('', port), handler)

        self.ip = socket.gethostbyname(socket.gethostname())
        self.port = self.server.server_port
        threading.Thread(target=self.open).start()

    def open(self):
        self.server.serve_forever()

    def close(self):
        self.server.shutdown()


def make_handler_class(html_root="web", data_root="output"):

    class Handler(BaseHTTPRequestHandler, object):

        def __init__(self, *args, **kwargs):
            self.html_root = os.path.abspath(html_root)
            self.data_root = os.path.abspath(data_root)
            self.path = None
            super(Handler, self).__init__(*args, **kwargs)

        def log_message(self, format_, *args):
            return

        def do_GET(self):
            args = None

            if self.path == "/":
                self.path = "index.html"

            if '?' in self.path:
                self.path, tmp = self.path.split('?', 1)
                args = urllib.parse.parse_qs(tmp)

            try:
                send_static_data = False
                send_binary_data = False

                if self.path.endswith(".html"):
                    mime_type = 'text/html'
                    send_static_data = True
                elif self.path.endswith(".htm"):
                    mime_type = 'text/html'
                    send_static_data = True
                elif self.path.endswith(".jpg"):
                    mime_type = 'image/jpg'
                    send_static_data = True
                elif self.path.endswith(".png"):
                    mime_type = 'image/png'
                    send_static_data = True
                elif self.path.endswith(".gif"):
                    mime_type = 'image/gif'
                    send_static_data = True
                elif self.path.endswith(".js"):
                    mime_type = 'application/javascript'
                    send_static_data = True
                elif self.path.endswith(".css"):
                    mime_type = 'text/css'
                    send_static_data = True
                else:
                    mime_type = 'application/octet-stream'
                    send_binary_data = True

                if send_static_data:
                    f = open(self.html_root + os.sep + self.path, 'rb')
                    self.send_response(200)
                    self.send_header('Content-Type', mime_type)
                    self.end_headers()
                    self.wfile.write(f.read())
                    f.close()
                elif send_binary_data:
                    data, is_gzipped = self.__get_data(self.path, args)
                    self.send_response(200)
                    self.send_header('Content-Type', mime_type)
                    self.send_header("Content-Length", len(data))
                    if is_gzipped:
                        self.send_header("Content-Encoding", 'gzip')
                    self.end_headers()
                    self.wfile.write(data)
                return

            except IOError:
                self.send_error(404, 'File Not Found: %s' % self.path)

        def __get_data(self, request, args):
            data = None
            frame = None
            layer = None
            is_gzipped = None

            if args is not None:
                if 'frame' in args and args['frame'] is not None:
                    if args['frame'][0] is not None:
                        try:
                            frame = int(args['frame'][0])
                        except ValueError:
                            frame = 0
                        if frame <= 0:
                            frame = None
                if 'layer' in args and args['layer'] is not None:
                    layer = str(args['layer'][0])

            # requesting frames
            if request == "/video":
                if frame is not None:
                    file_name = Handler.__frame_to_path(frame)
                    file_name = self.data_root + os.sep + "frames" + os.sep + file_name + ".png"
                    with open(file_name, "rb") as f:
                        data = f.read()
                    is_gzipped = False  # it is png-zipped, and not gzipped too

            # requesting optical flow
            elif request == "/motion":
                if frame is not None:
                    file_name = Handler.__frame_to_path(frame)
                    file_name = self.data_root + os.sep + "motion" + os.sep + file_name + ".of"
                    with open(file_name, "rb") as f:
                        data = f.read()
                    is_gzipped = True

            # requesting features
            elif request == "/features":
                if frame is not None:
                    file_name = Handler.__frame_to_path(frame)
                    file_name = self.data_root + os.sep + "features" + os.sep + file_name + ".feat_" + layer
                    with open(file_name, "rb") as f:
                        data = f.read()
                    is_gzipped = True

            # requesting filters
            elif request == "/filters":
                if frame is not None:
                    file_name = Handler.__frame_to_path(frame)
                    file_name = self.data_root + os.sep + "filters" + os.sep + file_name + ".fil_" + layer
                    with open(file_name, "rb") as f:
                        data = f.read()
                    is_gzipped = True

            # requesting options and details
            elif request == "/options":
                file_name = self.data_root + os.sep + "options.txt"
                with open(file_name, "r") as f:
                    data = f.read()
                data = str.encode(data)
                is_gzipped = False

            # requesting other stuff
            elif request == "/others":
                if frame is not None:
                    file_name = Handler.__frame_to_path(frame)
                    file_name = self.data_root + os.sep + "others" + os.sep + file_name + "_" + layer + ".txt"
                    with open(file_name, "r") as f:
                        data = f.read()
                    data = str.encode(data)
                    is_gzipped = False

            # unsupported data request
            if data is None:
                raise IOError()

            return data, is_gzipped

        @staticmethod
        def __frame_to_path(frame):
            files_per_folder = InputStream.files_per_folder
            f = frame - 1
            n_folder = int(f / files_per_folder) + 1
            n_file = (f + 1) - ((n_folder - 1) * files_per_folder)

            folder_name = format(n_folder, '08d')
            file_name = format(n_file, '03d')

            return folder_name + os.sep + file_name

    return Handler


if __name__ == '__main__':
    visualization_server = VisualizationServer(port=0, data_root=os.path.abspath(sys.argv[1]))
    out()
    out('[Visualization Server]')
    out('- IP: ' + str(visualization_server.ip))
    out('- Port: ' + str(visualization_server.port))
    out('- Data Root: ' + os.path.abspath(sys.argv[1]))
    out("- URL: http://" + str(visualization_server.ip) + ":" + str(visualization_server.port))
    out()
