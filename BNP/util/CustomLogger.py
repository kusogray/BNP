'''
Created on Jan 30, 2016

@author: yu1
'''

import time
import os
import subprocess
# Import smtplib for the actual sending function
import smtplib


# Import the email modules we'll need
from email.mime.text import MIMEText
from BNP.Resources import Config

def musicAlarm():
    if os.name == 'nt':
        os.startfile(Config.musicAlarmPath)
    else:
        opener = "open"
        subprocess.call([opener, Config.musicAlarmPath])
        
def info(*inputMsg):
    tmpStr=""
    for tmp in inputMsg:
        tmpStr+= str(tmp) 
    msg = "[logger] " + str(time.strftime("%Y-%m-%d")) + ' ' +str((time.strftime("%H:%M:%S"))) + " " + tmpStr
    print msg
    
    with open(Config.allTimeLogPath, "a") as myfile:
        myfile.write(msg +"\n")

def mail(mailTitle, *inputMsg):
    
    tmpStr=""
    for tmp in inputMsg:
        tmpStr+= str(tmp) 
    accPath = Config.mailAccPath

    file = open(accPath, 'r')
    accInfo = file.read()
    file.close() 
    
    infoList =  accInfo.split(',')

    me = infoList[1]
    you = infoList[3]
    
    msg = MIMEText(tmpStr)
    msg['Subject'] = mailTitle
    msg['From'] = me  
    msg['To'] = you 

    

    s=smtplib.SMTP_SSL()
    s.connect("smtp.googlemail.com",465)
    s.ehlo()
    s.login(infoList[1], infoList[2])
    s.sendmail(me, you, msg.as_string())
    s.quit()
    

if __name__ == '__main__':
    mail("Hoho New!!!", "DerDer!!")
    musicAlarm()