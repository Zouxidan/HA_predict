import csv
import os
import numpy as np
import pandas as pd
from PyQt5.Qt import *
from PyQt5 import QtWidgets, QtCore
import feature
import train_model


class Ui_HA_prediction(object):
    def setupUi(self, HA_prediction):
        HA_prediction.setObjectName("HA_prediction")
        HA_prediction.resize(1600, 800)
        self.horizontalLayoutWidget = QtWidgets.QWidget(HA_prediction)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1581, 771))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.line = QtWidgets.QFrame(self.horizontalLayoutWidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.textEdit = QtWidgets.QTextEdit(self.horizontalLayoutWidget)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.textEdit.setPlaceholderText('FASTA sequence')
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_4 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.line_3 = QtWidgets.QFrame(self.horizontalLayoutWidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout_2.addWidget(self.line_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.line_2 = QtWidgets.QFrame(self.horizontalLayoutWidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        self.tableView = QtWidgets.QTableView(self.horizontalLayoutWidget)
        self.tableView.setObjectName("tableView")
        self.verticalLayout_2.addWidget(self.tableView)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.pushButton_5 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_2.addWidget(self.pushButton_5)
        self.retranslateUi(HA_prediction)
        self.time = 0
        self.pushButton.clicked.connect(self.textEdit.clear) # type: ignore
        self.pushButton_3.clicked.connect(self.onClick_ButtonText)
        self.pushButton_4.clicked.connect(self.readtxt)
        self.pushButton_2.clicked.connect(self.wait)
        self.pushButton_2.clicked.connect(self.ui)
        self.pushButton_5.clicked.connect(self.save)
        QtCore.QMetaObject.connectSlotsByName(HA_prediction)

    def retranslateUi(self, HA_prediction):
        _translate = QtCore.QCoreApplication.translate
        HA_prediction.setWindowTitle(_translate("HA_prediction", "HA_prediction"))
        self.label.setText(_translate("HA_prediction", "Prediction sequence"))
        self.pushButton_2.setText(_translate("HA_prediction", "Submit"))
        self.pushButton_4.setText(_translate("HA_prediction", "File"))
        self.pushButton_3.setText(_translate("HA_prediction", "Example"))
        self.pushButton_5.setText(_translate("HA_prediction", "save"))
        self.pushButton.setText(_translate("HA_prediction", "Clear"))
        self.label_2.setText(_translate("HA_prediction", "Prediction result"))

    def wait_message(self):
        QMessageBox.information(self, "Successful", "Upload successful, please wait")

    def empty_message(self):
        QMessageBox.critical(self, "Empty", "Empty, please enter the sequence")

    def wait(self):
        test = self.textEdit.toPlainText()
        if not test:
            msg_box1 = QMessageBox(QMessageBox.Critical, "Empty", "Empty, please enter the sequence")
            msg_box1.exec_()
        else:
            msg_box2 = QMessageBox(QMessageBox.Information, "Successful", "Upload successful, please wait")
            msg_box2.exec_()


    def onClick_ButtonText(self):
        # 调用文本框设置普通文本
        self.textEdit.setPlainText( '>Q9E780\n'
                                    'MGSFIYKQLLTNSYTVELSDEIDAIGSEKTQNVTINPGPFAQTGYAPVEWGAGETNDSTTIEPVLDGPYQPTRFNPEIGYWILLAPETQGIVLETTNTTNKWFATILIEQDVVAESRTYTIFGKTESIQAENTSQTEWKFIDIIKTTQDGTYSQYGPLVLSTKLYGVMKYGGRLYAYIGHTPNATPGHYTIANYDTMEMSIFCEFYIMPRSQEAQCTEYINSGLPPIQNTRNIVPLSLSSRSIKYQKAQVNEDIIISKTSLWKEMQYNIDIIIRFKFNNSIIKSGGLGYKWLEIAFKPANYQYNYIRDGENITAHTTCSVNGVNEFSYNGGSLPTDFAISRYEVIKENSYVYVDYWDDSQAFRNMVYVRSLAANLNTVICNGGDYSFQVPVGQWPVMSGGAVSLQSAGVTLSTQFTDFVSLNSLRFRFSLAVESPPFSITRTRVSNLYGLPAANPNGGRDFYEILGRFSLISLVPSNDDYQTPIMNSVTVRQDLDRQLGELRDEFNALSQQIAMSQLIDLALLPLDMFSMFSGIKGSIDVARSMATKVMKKFRNSKLASSVSTLTDSLSDAASSLSRTSTIRSIGSSASAWTNISSQVDDVISSTSEISTQTSTISRRLRVKEIATQTEGMNFDDISAAVLKAKIDRSTQIDSNTLPDIVTEASEKFIPNRAYRVMDGDEVLEASTDGKFFAYKVETFDEVPFDVQKFADLVTDSPVISAIIDFKTLKNLNDNYGITKAQAFNLLRSDPRVLREFINQENPIIRNRIEQLILQCKL\n'
                                    '>P83956\n'
                                    'ANQTYFNFQRFEETN\n'
                                    '>P27169\n'
                                    'MAKLIALTLLGMGLALFRNHQSSYQTRLNALREVQPVELPNCNLVKGIETGSEDLEILPNGLAFISSGLKYPGIKSFNPNSPGKILLMDLNEEDPTVLELGITGSKFDVSSFNPHGISTFTDEDNAMYLLVVNHPDAKSTVELFKFQEEEKSLLHLKTIRHKLLPNLNDIVAVGPEHFYGTNDHYFLDPYLQSWEMYLGLAWSYVVYYSPSEVRVVAEGFDFANGINISPDGKYVYIAELLAHKIHVYEKHANWTLTPLKSLDFNTLVDNISVDPETGDLWVGCHPNGMKIFFYDSENPPASEVLRIQNILTEEPKVTQVYAENGTVLQGSTVASVYKGKLLIGTVFHKALYCEL\n'
                                    '>Q9GZV9\n'
                                    'MLGARLRLWVCALCSVCSMSVLRAYPNASPLLGSSWGGLIHLYTATARNSYHLQIHKNGHVDGAPHQTIYSALMIRSEDAGFVVITGVMSRRYLCMDFRGNIFGSHYFDPENCRFQHQTLENGYDVYHSPQYHFLVSLGRAKRAFLPGMNPPPYSQFLSRRNEIPLIHFNTPIPRRHTRSAEDDSERDPLNVLKPRARMTPAPASCSQELPSAEDNSPMASDPLGVVRGGRVNTHAGGTGPEGCRPFAKFI\n'
                                    '>P85004\n'
                                    'ESGINLQGDALANN\n'
                                    '>Q9UK05\n'
                                   'MCPGALWVALPLLSLLAGSLQGKPLQSWGRGSAGGNAHSPLGVPGGGLPEHTFNLKMFLENVKVDFLRSLNLSGVPSQDKTRVEPPQYMIDLYNRYTSDKSTTPASNIVRSFSMEDAISITATEDFPFQKHILLFNISIPRHEQITRAELRLYVSCQNHVDPSHDLKGSVVIYDVLDGTDAWDSATETKTFLVSQDIQDEGWETLEVSSAVKRWVRSDSTKSKNKLEVTVESHRKGCDTLDISVPPGSRNLPFFVVFSNDHSSGTKETRLELREMISHEQESVLKKLSKDGSTEAGESSHEEDTDGHVAAGSTLARRKRSAGAGSHCQKTSLRVNFEDIGWDSWIIAPKEYEAYECKGGCFFPLADDVTPTKHAIVQTLVHLKFPTKVGKACCVPTKLSPISVLYKDDMGVPTLKYHYEGMSVAECGCR\n'
                                   '>Q04916\n'
                                   'MLTYLRREWQSFGETVTIKNTFNAQEDNNQSGRKTDNRPVKTEGRYCYKADVNRSKYYHDVQGFSLGQSDLHIDPTQFIMYSGTISNGISYVNQAPSCVQLSLKFTPGNSSLIEDLHIEPYKVEVLKIEHVGNVSRATLLSDIVSLSIAQKKLLLYGFTQLGIQGLTGDVVSVETKRIPTPTQTNLLTIEDSMQCFTWDMNCANVRSTKQDSRLIIYEQEDGFWKIVTETLSIKVKPYFKAYGTMGGAFKNWLVDSGFEKYQHDLAYVRDGVTVNAHTITYVNPSGKAGLQQDWRPATDYNGQITVLQPGDGFSVWYYEDKWQINQAIYAKNFQSDTRAQGYLENVGTLKFKMNYIPAFAEIRNKPGKVNYAYLNGGFAQVDASGYTGMSIILNFVCTGERFYASDNNSRVDNKITPFISYIGDYYTLSGGDFYRQGCCAGFAAGYDDVSPEHGITVSYTVMKPSDPDFITGGENYGESITSDLEVSIRNLQDQINSIIAEMNIQQVTSAVFTAITNLGELPGLFSNITKVFSKTKEALSKLKSRKKTSPMPIAATSIIDKTTVDVPNLTIVNKMPEEYELGIIYNSMRTKKLIEQKKHDFSTFTVATEVKLPYISKATNFSDQFMTSISSRGITIGKSDIIQYDPMNNILSAMNRKNAQIINYKIDPDLAHEVLSQMSTNATRSLFSLNVRKQLHINNSFDTPTYGQLVERILDDGQLLDILGKLNPNSVEELFSEFLHRIQHQLREY'
                                    '>P20142\n'
                                    'MKWMVVVLVCLQLLEAAVVKVPLKKFKSIRETMKEKGLLGEFLRTHKYDPAWKYRFGDLSVTYEPMAYMDAAYFGEISIGTPPQNFLVLFDTGSSNLWVPSVYCQSQACTSHSRFNPSESSTYSTNGQTFSLQYGSGSLTGFFGYDTLTVQSIQVPNQEFGLSENEPGTNFVYAQFDGIMGLAYPALSVDEATTAMQGMVQEGALTSPVFSVYLSNQQGSSGGAVVFGGVDSSLYTGQIYWAPVTQELYWQIGIEEFLIGGQASGWCSEGCQAIVDTGTSLLTVPQQYMSALLQATGAQEDEYGQFLVNCNSIQNLPSLTFIINGVEFPLPPSSYILSNNGYCTVGVEPTYLSSQNGQPLWILGDVFLRSYYSVYDLGNNRVGFATAA\n'
                                    '>D2YVI2\n'
                                    'MGRFLLVTLSLLVMAFFLNGANSCCCPQDWLPRNGFCYKVFNDLKTWDDAEMYCRKFKPGCHLASLHSNADAVEFSEYITDYLTGQGHVWIGLRDTKKKYIWEWTDRSRTDFLPWRKDQPDHFNNEEFCVEIVNFTGYLQWNDDSCTALRPFLCQCKY\n'
                                    '>B3EWR1\n'
                                    'MTFAKQSCFNSIILLSIATSYFKIGHKISELGNRIEKMTTFLIKHKASGKFLHPKGGSSNPANDTNLVLHSDIHERMYFQFDVVDERWGYIKHAASGKIVHPLGGKADPPNETKLVLHQDRHDRALFAMDFFNDNIIHKAGKYVHPKGGSTNPPNETLTVMHGDKHGAMEFIFVSPKNKDKRVLVYV\n'
                                    '>P31614\n'
                                    'MGCMCIAMAPRTLLLLIGCQLVFGFNEPLNIVSHLNDDWFLFGDSRSDCTYVENNGHPKLDWLDLDPKLCNSGRISAKSGNSLFRSFHFIDFYNYSGEGDQVIFYEGVNFSPSHGFKCLAYGDNKRWMGNKARFYARVYEKMAQYRSLSFVNVSYAYGGNAKPTSICKDKTLTLNNPTFISKESNYVDYYYESEANFTLQGCDEFIVPLCVFNGHSKGSSSDPANKYYTDSQSYYNMDTGVLYGFNSTLDVGNTVQNPGLDLTCRYLALTPGNYKAVSLEYLLSLPSKAICLRKPKSFMPVQVVDSRWNSTRQSDNMTAVACQLPYCFFRNTSADYSGGTHDVHHGDFHFRQLLSGLLYNVSCIAQQGAFVYNNVSSSWPAYGYGHCPTAANIGYMAPVCIYDPLPVILLGVLLGIAVLIIVFLMFYFMTDSGVRLHEA'
                                   )


    def readtxt(self):
        # 实例化QFileDialog
        dig = QFileDialog()
        # 设置可以打开任何文件
        dig.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        dig.setFilter(QDir.Files)

        if dig.exec_():
            # 接受选中文件的路径，默认为列表
            filenames = dig.selectedFiles()
            # 列表中的第一个元素即是文件路径，以只读的方式打开文件
            f = open(filenames[0], 'r')

            with f:
                # 接受读取的内容，并显示到多行文本框中
                data = f.read()
                self.textEdit.setText(data)


    def ui(self):
        self.time = self.time+1
        if self.time>1:
            self.model.clear()
        self.th = thread(self)
        self.th.signal.connect(self.handleDisplay)
        self.th.start()

    def handleDisplay(self, data):
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Name', 'Sequence', 'Result', 'Score'])
        model = self.tableView.model()
        pre_result = data
        for i in range(0,len(data)):
            row = pre_result[i]
            for j in range(0,len(row)):
                column = row[j]
                item = QStandardItem(str(column))
                self.model.setItem(i, j, item)
        self.tableView.setModel(self.model)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)



    def save(self):
        file_data = []
        #
        row = self.model.rowCount()
        for i in range(0,row):
            index1 = self.model.index(i, 0)
            index2 = self.model.index(i, 1)
            index3 = self.model.index(i, 2)
            index4 = self.model.index(i, 3)
            name = self.model.data(index1)
            seq = self.model.data(index2)
            re = self.model.data(index3)
            score = self.model.data(index4)
            res = [name,seq,re,score]
            file_data.append(res)
            file_res = pd.DataFrame(file_data,columns=['Name','sequence','result','score'])
        self.fname, ftype = QFileDialog.getSaveFileName(self, 'save file', './', 'CSV Files (*.csv)')
        if self.fname:
            if os.path.exists(os.path.abspath(self.fname)):
                with open(self.fname, 'w') as f:
                    f.truncate(0)
                    csv_writer = csv.writer(f, delimiter=',')
                    csv_writer.writerow(['Name', 'sequence', 'result', 'score'])
                    for i in range(0, file_res.shape[0]):
                        csv_writer.writerow(file_res.iloc[i, :])
            with open(self.fname,'w',newline="") as csv_file:
                csv_writer = csv.writer(csv_file,delimiter=',')
                csv_writer.writerow(['Name', 'sequence', 'result', 'score'])
                for i in range(0,file_res.shape[0]):
                    csv_writer.writerow(file_res.iloc[i,:])



class thread(QThread):
    signal = pyqtSignal(list)

    def __init__(self,test):
        super(thread, self).__init__()
        self.test = test

    def run(self):
        data = self.test.textEdit.toPlainText()
        records = data.split('>')[1:]
        fasta_sequences = []
        result1 = pd.DataFrame([],columns=['Entry', 'Sequence'])
        i = 0
        for fasta in records:
            names = fasta.split('\n')[0]
            seq = fasta.split('\n')[1]
            re1 = [names,seq]
            result1.loc[result1.shape[0]] = re1
            i = i + 1
            array = fasta.split('\n')
            header, sequence = array[0].split()[0], array[1].split()[0]
            fasta_sequences.append([header, sequence])
            fasta_data = pd.DataFrame(fasta_sequences, columns=['Entry', 'Sequence'])
            fasta_data['Entry'] = ['>%s|1|testing' % i for i in fasta_data["Entry"]]
            fasta_data['Sequence'] = ['%s' % i for i in fasta_data["Sequence"]]

        with open(r'predict_data.txt', 'w') as f:
            for fl in range(0, fasta_data.shape[0]):
                f.write(fasta_data.iloc[fl, 0] + '\n')
                f.write(fasta_data.iloc[fl, 1] + '\n')

        train_data = feature.get_feature('./data/train_data.txt')
        print('*Training set feature extraction completed...\n')
        pre_data = feature.get_feature('./data/predict_data.txt')
        print('*Predictive set feature extraction completed...\n')
        data_train, data_predict = train_model.feature_subset(train_data, pre_data)
        y_proba_valid_all, y_proba_pre_all, y_original_label_valid, y_original_label_pre = train_model.get_result(data_train, data_predict)
        pred_y = np.where(np.array(y_proba_pre_all) >= 0.5, 'HA', np.array(y_proba_pre_all))
        pred_y = np.where(np.array(y_proba_pre_all) < 0.5, 'non-HA', pred_y)
        a = 0
        result2 = pd.DataFrame([],columns=['result'])
        for y in pred_y:
            result2.loc[result2.shape[0]] = y
            a = a + 1

        result3 = pd.DataFrame([], columns=['probability'])
        probability = np.array(y_proba_pre_all)
        b = 0
        for pro in probability:
            if np.array(pro) >= 0.5:
                pro = float(pro)*100
            else:
                pro = float(pro)*100
            result3.loc[result3.shape[0]] = ('%0.2f'%pro)
            b = b + 1

        result_all = pd.concat([result1,result2,result3],axis=1)
        result_list = result_all.values.tolist()
        self.signal.emit(list(result_list))
