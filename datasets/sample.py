from torchvision import io

import moma_api


def main():
  data_dir = '/home/ubuntu/datasets/MOMA'
  api = moma_api.get_moma_api(data_dir, 'trim')

  trim_ids = ['20201117105359',
             '20201117133848',
             '20201203194312',
             '20201203194523',
             '20201203194540',
             '20201203194554',
             '20201203201129',
             '20201203201205',
             '20201203201522',
             '20201203201541',
             '20201203212956',
             '20201203213045',
             '20201203220233',
             '20201203220314']

  for trim_id in trim_ids:
    trim_ann = api.anns[trim_id]

    video_path = api.get_video_path(trim_id, sample=True)
    video = io.read_video(video_path, pts_unit='sec')[0]

    print(video.shape, len(trim_ann['ags']))

    break


if __name__ == '__main__':
  main()