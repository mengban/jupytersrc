#北邮网关登录
#开机自动运行
import urllib.request, urllib.error, urllib.parse
import urllib.request, urllib.parse, urllib.error
import re
import os
import time

class Loginer():
    def __init__(self, username, password):
        self.loginUrl = 'http://10.3.8.211/'
        self.username = username
        self.password = password
        self.openner = urllib.request.build_opener()

    def login(self):
        postdata = {
            'DDDDD': self.username,
            'upass': self.password,
            'savePWD': 0,
            '0MKKey': ''
        }
        postdata = urllib.parse.urlencode(postdata).encode(encoding="UTF8")
        myRequest = urllib.request.Request(url=self.loginUrl, data=postdata)
        
        result = self.openner.open(myRequest).read()
        
        unicodePage = result.decode('gb2312')
        #print(unicodePage)
        # print unicodePage
        msg = re.findall('<title>(.*?)</title>', unicodePage)[0]
        print('This is msg:',msg)
        if msg== '登录成功窗':
            print('user: ', self.username, '******login success!******')
        else:
            print('user: ', self.username, '******login fail!******')


def main():
   # exit_code=os.system('ping -n 1 -w 100 10.3.8.211') #-n 次数 -w 超时时间
   # while(exit_code):
       # time.sleep(1)
       # print('请检测网线是否插好或者已连接WiFi...')
       # exit_code=os.system('ping -n 1 -w 100 10.3.8.211')
    l = Loginer('2017110615', '132512') #用户在这里输入自己的学号与密码
    l.login()


if __name__ == '__main__':
    main()
