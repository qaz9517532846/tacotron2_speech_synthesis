import csv
import io

read_text = open('D:/Taipei_Tech/Deep_Learning_audio/tacotron2/filelists/ljs_audio_text_train_filelist.txt', encoding="utf-8")
output_txt = io.open("filelists/train_list.txt", "a", encoding="utf-8")

for line in read_text:
      line_result = line.split('|', 1)
      print('Processing ID:' + line_result[0])
      result_text = line_result[0] + '\n'
      output_txt.writelines(result_text)
print('Finish')