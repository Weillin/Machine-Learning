import xlrd #读取excel
import xlwt #写入excel

data = xlrd.open_workbook('ces.xlsx')

sheet = data.sheet_names()
print(sheet)

sheet1 = data.sheet_names()[0]
print(sheet1)

# li_1 = []
# li_2 = []
# for j in li_set:
# li_1.append(j)
# li_2.append(li.count(j))
# print(li_1)
# print(li_2)