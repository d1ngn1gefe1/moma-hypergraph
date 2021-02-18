import cv2
from datasets.moma_api import get_momaapi
from pprint import pprint


class Visualizer:
  def __init__(self, api, trimmed_video_ids):
    self.api = api
    self.trimmed_video_ids = trimmed_video_ids

  def draw_image(self, image, graph):
    font = cv2.FONT_HERSHEY_SIMPLEX

    for actor in graph['actors']:
      color = (0, 255, 0)
      x1, y1, w, h = [round(b) for b in actor['bbox']]
      cname = self.api.actor_cnames[actor['actor_cid']]
      text = '{}: {}'.format(cname, actor['instance_id'])

      cv2.rectangle(image, (x1, y1), (x1+w, y1+h), color, 3)
      cv2.putText(image, text, (x1+10, y1+h-10), font, 1, color, 2)

    for object in graph['objects']:
      color = (255, 0, 0)
      x1, y1, w, h = [round(b) for b in object['bbox']]
      cname = self.api.object_cnames[object['object_cid']]
      text = '{}: {}'.format(cname, object['instance_id'])

      cv2.rectangle(image, (x1, y1), (x1+w, y1+h), color, 3)
      cv2.putText(image, text, (x1+10, y1+h-10), font, 1, color, 2)

    text = []
    color = (0, 0, 255)
    for atomic_action in graph['atomic_actions']:
      cname = self.api.atomic_action_cnames[atomic_action['atomic_action_cid']]
      text.append('{}: {}'.format(cname, atomic_action['actor_instance_ids']))

    cv2.putText(image, ', '.join(text), (10, 40), font, 1, color, 2)

    text = []
    color = (0, 0, 255)
    for relationship in graph['relationships']:
      cname = self.api.relationship_cnames[relationship['relationship_cid']]
      text.append('{}: {}'.format(cname, relationship['description']))

    cv2.putText(image, ', '.join(text), (10, 80), font, 1, color, 2)

    return image

  def visualize(self):
    i = 0
    while True:
      video_path = self.api.get_video_path(self.trimmed_video_ids[i])
      graphs = self.api.get_annotation(self.trimmed_video_ids[i])['graphs']

      cap = cv2.VideoCapture(video_path)
      images = []
      while cap.isOpened():
        ret, image = cap.read()
        if ret:
          images.append(image)
        else:
          break

      assert len(graphs) == len(images)

      print(self.trimmed_video_ids[i], video_path, len(graphs), len(images), images[0].shape)
      # pprint(annotation)

      j = 0
      while True:
        image = self.draw_image(images[j], graphs[j])
        image = cv2.resize(image, None, fx=0.5, fy=0.5)

        pprint(graphs[j])
        print('\n\n')

        cv2.imshow('image', image)
        key = cv2.waitKey(0)

        if key & 0xFF == ord('a'):    # previous frame
          j = max(0, j-1)
        elif key & 0xFF == ord('d'):  # next frame
          j = min(len(graphs)-1, j+1)
        elif key & 0xFF == ord('w'):  # previous video
          i = max(0, i-1)
          break
        elif key & 0xFF == ord('s'):  # next video
          i = min(len(self.trimmed_video_ids)-1, i+1)
          break


def main():
  dataset_dir = '/Users/alanzluo/Documents/moma'
  api = get_momaapi(dataset_dir, 'trimmed_video')

  trimmed_video_ids = ['20201203210825']

  visualizer = Visualizer(api, trimmed_video_ids)
  visualizer.visualize()


if __name__ == '__main__':
  main()
