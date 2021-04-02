import xlsxwriter

def create(y_predict):
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook('output.xlsx')
    worksheet = workbook.add_worksheet()


    # Start from the first cell. Rows and columns are zero indexed.
    row = 1
    col = 0

    # Iterate over the data and write it out row by row.
    worksheet.write(0,0,'SECTION')
    for y in y_predict:
        worksheet.write(row, col, y)
        row += 1


    workbook.close()