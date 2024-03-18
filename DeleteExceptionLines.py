import csv

def save_cleaned_file(original_file_path, new_file_path, encoding='gb2312'):
    with open(original_file_path, 'rb') as infile, open(new_file_path, 'w', encoding=encoding, newline='') as outfile:
        writer = csv.writer(outfile)
        for binary_line in infile:
            try:
                # 尝试用gb2312解码，如果成功，则进一步处理
                line = binary_line.decode(encoding)
                # 使用csv模块正确处理引号和逗号
                reader = csv.reader([line])
                for row in reader:
                    writer.writerow(row)
            except UnicodeDecodeError:
                # 如果解码失败，就跳过这一行
                continue

# 使用之前的函数处理文件
file_path = './DatasetAll.csv'
new_file_path = './Cleaned_DatasetAll.csv'
save_cleaned_file(file_path, new_file_path, encoding='gb2312')

print(f"已删除无法使用gb2312编码读取的行，并保存了清洗后的数据到 {new_file_path}")
