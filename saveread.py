"""
In MathY, matrix can be saved as .csv file.
read_mat and write_mat is for .csv file, not for .mat file!
For .mat file, use "from scipy.io import loadmat" and loadmat(filename)
"""
from linalg import matrix_shape
import time

def read_matrix(filename,datatype = float):
    path = "data_save/" + filename
    with open(path, 'r',encoding="utf-8-sig") as file:
        contents = file.readlines()
        mat = []
        for row in contents:            
            row = row.strip('\n')
            row = row.split(',')
            row_list = []
            for item in row:
                row_list.append(datatype(item))
            mat.append(row_list)
    print("%d rows, %d columns loaded."%(len(mat),len(mat[0])))
    return mat

def save_matrix(data,filename):
    m,n = matrix_shape(data) # from linalg import matrix_shape
    path = "data_save/"+filename
    file = open(path,'w')
    for i in range(m):
        s = ''
        for j in range(n):
            s = s + str(data[i][j])#去除[],这两行按数据不同，可以选择
            if j != n-1:
                s = s+','   #去除单引号，逗号，每行末尾追加换行符
        s = s + '\n'
        file.write(s)

    file.close()
    print("%d rows, %d columns saved."%(m,n))

def readmat(filename,datatype = float):
    return read_matrix(filename,datatype)


def note(text):
    with open('data_save/notes.txt', 'a+') as f:
        localtime = time.asctime( time.localtime(time.time()) )+':\n'
        f.write(localtime+text+'\n\n') 

def show_note():
    with open('data_save/notes.txt', 'r') as f:
        print(f.read())
if __name__ == "__main__":
    A = randmat(4,5)
    save_matrix(A,"matrix")
    B = read_matrix("matrix")
    print_matrix(B)