from Scraping_specific_url import read_ods
from Scraping_specific_url import url_Nikkei


now = datetime.now()
today = now.strftime("%Y-%m-%d")
next_day = nextDay(today)
next_day = next_day[-2:] + '/' + next_day[-5:-3] + '/' + next_day[0:4]


agent = Random_agent_generator()  #generate a random agent user for the scraping
opts = Options()
opts.add_argument("User-agent= %s" %agent)
driver = webdriver.Chrome(chrome_options=opts)

driver.get("https://google.com")

driver.execute_script("window.open('');")
driver.switch_to.window(driver.window_handles[1])
driver.get("https://www.zonebourse.com/bourse/agenda/financier/")


element = driver.find_element_by_xpath('//*[@id="zbsFTp"]/div[2]/div[1]/select[1]')
element.click()
element = driver.find_element_by_xpath('//*[@id="listePays"]/ul[1]/li[1]/label[1]')
element.click()
time.sleep(1)
element = driver.find_element_by_xpath('//*[@id="sub_countries_all"]/li[6]/label[1]')
element.click()
time.sleep(1)
element = driver.find_element_by_xpath('//*[@id="sub_countries_6"]/li[14]/label')
element.click()
time.sleep(1)




soup_level1=BeautifulSoup(driver.page_source, 'html.parser')

dates_b = soup_level1.findAll('td','pleft10 pright10')  #select the tickers
tickers_b = soup_level1.findAll("a") #select the tickers
tickers_all = []
for i in range(0,len(tickers_b)):
    temp = str(tickers_b[i]).split()
    for j in range(0,len(temp)):
        if(temp[j] == 'at="1"' or temp[j] == 'at="0"'):
            tickers_all.append(tickers_b[i].text)


#we create the tickers for tomorrow
cpt = 0
tomorrow_list = []
for i in range(0,len(tickers_all)):

    tickers_date = dates_b[i].div["title"].split()
    for j in range(0,len(tickers_date)):#we locate the true date
        if(tickers_date[j][-5:] == '/2018'):
            true_date = tickers_date[j]
    if(true_date == next_day):
        tomorrow_list.append([tickers_all[cpt],true_date])
        cpt = cpt + 1
      
        
'https://svc.qri.jp/jpx/english/kbopm/'



correspondance_df = read_ods(filename='correspondance_list_NIKKEI.ods')
url_togo = []
for i in range(0,len(tomorrow_list)):
    for j in range(0,len(correspondance_df)):
        
        url_ref = ""
        if(correspondance_df['Name'][j].split()[0] + " " + correspondance_df['Name'][j].split()[1]  == tomorrow_list[i][0].split()[0]):
            url_ref = correspondance_df['Code'][j]
            url_togo.append([url_euronext(url_ref),correspondance_df['Name'][j]])
            locator = j
        
        print('to make sure this is the good stock: we have %(s)s -vs- %(n)s' % {'n': tomorrow_list[i], 's': correspondance_df['Name'][j]})
        
     




