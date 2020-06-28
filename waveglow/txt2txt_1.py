import csv
import io

output_txt = io.open("filelists/train_list.txt", "a", encoding="utf-8")

for line in range(1, 3119):
      print('Processing ID:' + str(line))
      result_text = 'train/' + str(line) + '.wav' + '\n'
      output_txt.writelines(result_text)
print('Finish')