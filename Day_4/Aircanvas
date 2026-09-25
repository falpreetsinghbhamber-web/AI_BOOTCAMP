#canvas
import cv2, numpy as np
from base64 import b64decode, b64encode
from IPython.display import display, Javascript
from google.colab.output import eval_js

def video_stream():
  display(Javascript('''
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
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 320, 240);
          result = captureCanvas.toDataURL('image/jpeg', 0.6);
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
      video.style.transform = 'scaleX(-1)';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia({video: true});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.style.transform = 'scaleX(-1)';
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);

      const instruction = document.createElement('div');
      instruction.innerHTML = '<span style="color:red;font-weight:bold;">Click here or on the video to STOP</span>';
      instruction.onclick = () => { shutdown = true; };
      div.appendChild(instruction);

      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 320;
      captureCanvas.height = 240;
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
  '''))

def video_frame(label, overlay):
  return eval_js('stream_frame("{}", "{}")'.format(label, overlay))

def js_to_image(js_reply):
  img_bytes = b64decode(js_reply.split(',')[1])
  return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)

# Blue range in HSV
LOWER_COLOR = np.array([95, 80, 50])
UPPER_COLOR = np.array([140, 255, 255])

kernel = np.ones((3, 3), np.uint8)
canvas = np.zeros((240, 320, 4), dtype=np.uint8)   # transparent ink layer

prev_point = None      # last point that ink was drawn to
smooth = None          # smoothed pen position
ALPHA = 0.5            # lower = smoother but laggier (try 0.3 to 0.7)
MAX_JUMP = 80          # ignore sudden jumps bigger than this (pixels)
MISS_LIMIT = 3         # frames the pen may vanish before the line breaks
misses = 0

video_stream()
label = 'Wave a BLUE marker'
overlay = ''

while True:
  reply = video_frame(label, overlay)
  if not reply:                                    # click the video to stop
    break

  frame = js_to_image(reply['img'])

  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove speckles
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill holes

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  raw_point = None
  area = 0
  if contours:
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area > 100:
      M = cv2.moments(largest)
      if M['m00'] != 0:
        raw_point = (M['m10'] / M['m00'], M['m01'] / M['m00'])

  curr_point = None
  if raw_point is not None:
    misses = 0
    # exponential smoothing removes jitter
    if smooth is None:
      smooth = raw_point
    else:
      smooth = (ALPHA * raw_point[0] + (1 - ALPHA) * smooth[0],
                ALPHA * raw_point[1] + (1 - ALPHA) * smooth[1])
    curr_point = (int(smooth[0]), int(smooth[1]))

    if prev_point is not None:
      dist = np.hypot(curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
      if dist < MAX_JUMP:                          # skip false jumps
        cv2.line(canvas, prev_point, curr_point, (0, 0, 255, 255), 3, cv2.LINE_AA)
    prev_point = curr_point
  else:
    misses += 1
    if misses > MISS_LIMIT:                        # pen really gone: lift the pen
      prev_point = None
      smooth = None

  layer = canvas.copy()
  if curr_point:
    cv2.circle(layer, curr_point, 6, (0, 255, 255, 255), -1, cv2.LINE_AA)

  _, png = cv2.imencode('.png', layer)
  overlay = 'data:image/png;base64,' + b64encode(png.tobytes()).decode()
  label = 'Blue area: {} - {}'.format(int(area), 'tracking' if curr_point else 'not found')

print("Air Canvas closed.")
