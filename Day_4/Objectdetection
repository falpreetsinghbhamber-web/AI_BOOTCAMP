#object detection
!pip install -q ultralytics


import cv2, io, numpy as np
from base64 import b64decode, b64encode
from PIL import Image as PILImage
from IPython.display import display, Javascript
from google.colab.output import eval_js
from ultralytics import settings
settings.update({"sync": False})
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# ---- browser side: opens the camera ONCE and keeps it live ----
def video_stream():
  js = Javascript('''
    var video, div = null, stream, captureCanvas, imgElement, labelElement;
    var pendingResolve = null, shutdown = false;

    function removeDom() {
      stream.getVideoTracks()[0].stop();
      video.remove(); div.remove();
      video = null; div = null; stream = null; imgElement = null;
      captureCanvas = null; labelElement = null;
    }
    function onAnimationFrame() {
      if (!shutdown) window.requestAnimationFrame(onAnimationFrame);
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8);
        }
        var lp = pendingResolve; pendingResolve = null; lp(result);
      }
    }
    async function createDom() {
      if (div !== null) return stream;
      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);

      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status: </span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'Starting...';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);

      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia({video: true});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);

      const instruction = document.createElement('div');
      instruction.innerHTML = '<span style="color:red;font-weight:bold;">Click here or on the video to STOP</span>';
      instruction.onclick = () => { shutdown = true; };
      div.appendChild(instruction);

      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640;
      captureCanvas.height = 480;
      window.requestAnimationFrame(onAnimationFrame);
      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) { removeDom(); shutdown = false; return ''; }
      stream = await createDom();
      if (label != "") labelElement.innerHTML = label;
      if (imgData != "") {
        var r = video.getClientRects()[0];
        imgElement.style.top = r.top + "px";
        imgElement.style.left = r.left + "px";
        imgElement.style.width = r.width + "px";
        imgElement.style.height = r.height + "px";
        imgElement.src = imgData;
      }
      var result = await new Promise(function(resolve) { pendingResolve = resolve; });
      shutdown = false;
      return {'img': result};
    }
  ''')
  display(js)

def video_frame(label, overlay):
  return eval_js('stream_frame("{}", "{}")'.format(label, overlay))

# ---- helpers ----
def js_to_image(js_reply):
  img_bytes = b64decode(js_reply.split(',')[1])
  return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)

def overlay_to_data_url(rgba):
  buf = io.BytesIO()
  PILImage.fromarray(rgba, 'RGBA').save(buf, format='png')
  return 'data:image/png;base64,' + b64encode(buf.getvalue()).decode()

# ---- main loop (your detection + counting logic) ----
video_stream()
label = 'Starting...'
overlay = ''

while True:
  reply = video_frame(label, overlay)
  if not reply:                       # you clicked stop
    break

  frame = js_to_image(reply['img'])

  results = model(frame, verbose=False)
  boxes = results[0].boxes
  detected_count = len(boxes)

  # transparent layer that sits on top of the live video
  layer = np.zeros((480, 640, 4), dtype=np.uint8)
  for b in boxes:
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    name = model.names[int(b.cls[0])]
    conf = float(b.conf[0])
    cv2.rectangle(layer, (x1, y1), (x2, y2), (0, 255, 0, 255), 2)
    cv2.putText(layer, f'{name} {conf:.2f}', (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0, 255), 2, cv2.LINE_AA)

  cv2.putText(layer, f'Total Objects Counted: {detected_count}', (20, 50),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 255), 2, cv2.LINE_AA)

  overlay = overlay_to_data_url(layer)
  label = f'Objects: {detected_count}'

print("Camera feed closed successfully.")
