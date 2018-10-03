from Scraping_specific_url import read_ods
from Scraping_specific_url import url_euronext





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
#https://finviz.com/screener.ashx?v=111&f=earningsdate_tomorrowbefore&o=-marketcap&r=21


element = driver.find_element_by_xpath('//*[@id="zbsFTp"]/div[2]/div[1]/select[1]')
element.click()
element = driver.find_element_by_xpath('//*[@id="listePays"]/ul[1]/li[1]/label[1]')
element.click()
time.sleep(1)
element = driver.find_element_by_xpath('//*[@id="sub_countries_all"]/li[3]/label[1]')
element.click()
time.sleep(1)
element = driver.find_element_by_xpath('//*[@id="sub_countries_3"]/li[1]/label[1]')
element.click()
time.sleep(1)
element = driver.find_element_by_xpath('//*[@id="sub_countries_3"]/li[18]/label[1]')
element.click()
time.sleep(1)
element = driver.find_element_by_xpath('//*[@id="sub_countries_3"]/li[22]/label[1]')
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
    

#we create the correspondance for the ticker



correspondance_df = read_ods(filename='correspondance_list_EURONEXT.ods')
url_togo = []
for i in range(0,len(tomorrow_list)):
    for j in range(0,len(correspondance_df)):
        
        url_ref = ""
        if(correspondance_df['Name'][j] == "SOCIETE BIC"): 
            if(len(tomorrow_list[i][0].split()) > 1):
                if(correspondance_df['Name'][j].split()[1] == tomorrow_list[i][0].split()[1]):
                    url_ref = correspondance_df['Contract'][j]
                    url_togo.append([url_euronext(correspondance_df['LOCATION'][j],url_ref),correspondance_df['Name'][j]])
                    locator = j
        elif(correspondance_df['Name'][j] == "AIR FRANCE-KLM"): 
            if(len(tomorrow_list[i][0].split()) > 1):
                if(correspondance_df['Name'][j].split()[1] == tomorrow_list[i][0].split()[1]):
                    url_ref = correspondance_df['Contract'][j] 
                    url_togo.append([url_euronext(correspondance_df['LOCATION'][j],url_ref),correspondance_df['Name'][j]])
                    locator = j
        elif(correspondance_df['Name'][j] == "AIR LIQUIDE SA"): 
            if(len(tomorrow_list[i][0].split()) > 1):
                if(correspondance_df['Name'][j].split()[1] == tomorrow_list[i][0].split()[1]):
                    url_ref = correspondance_df['Contract'][j]       
                    url_togo.append([url_euronext(correspondance_df['LOCATION'][j],url_ref),correspondance_df['Name'][j]])
                    locator = j
        elif(correspondance_df['Name'][j] == "DEUTSCHE BANK AG"): 
            if(len(tomorrow_list[i][0].split()) > 1):
                if(correspondance_df['Name'][j].split()[1] == tomorrow_list[i][0].split()[1]):
                    url_ref = correspondance_df['Contract'][j]    
                    url_togo.append([url_euronext(correspondance_df['LOCATION'][j],url_ref),correspondance_df['Name'][j]])
                    locator = j
        elif(correspondance_df['Name'][j] == "DEUTSCHE POST AG"): 
            if(len(tomorrow_list[i][0].split()) > 1):
                if(correspondance_df['Name'][j].split()[1] == tomorrow_list[i][0].split()[1]):
                    url_ref = correspondance_df['Contract'][j]       
                    url_togo.append([url_euronext(correspondance_df['LOCATION'][j],url_ref),correspondance_df['Name'][j]])
                    locator = j
        elif(correspondance_df['Name'][j] == "DEUTSCHE TELEKOM AG"): 
            if(len(tomorrow_list[i][0].split()) > 1):
                if(correspondance_df['Name'][j].split()[1] == tomorrow_list[i][0].split()[1]):
                    url_ref = correspondance_df['Contract'][j]
                    url_togo.append([url_euronext(correspondance_df['LOCATION'][j],url_ref),correspondance_df['Name'][j]])
                    locator = j
        
        elif(correspondance_df['Name'][j].split()[0] == tomorrow_list[i][0].split()[0]):
            url_ref = correspondance_df['Contract'][j]
            url_togo.append([url_euronext(correspondance_df['LOCATION'][j],url_ref),correspondance_df['Name'][j]])
            locator = j
        
        print('to make sure this is the good stock: we have %(s)s -vs- %(n)s' % {'n': tomorrow_list[i], 's': correspondance_df['Name'][j]})
        
     



for k in range(0,len(url_togo)):
    

    url = url_togo[k][0]    
    driver.get(url)

 
    soup_level1=BeautifulSoup(driver.page_source, 'html.parser')
    
    
    table = soup_level1.find('div',{"class": "call-put-table"})  #select the tickers
    rows = table.find_all('tr')
    
    data = []
    for row in rows[1:]:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])
        
    result = pd.DataFrame(data, columns=['Settl.', 'Open Interest', 'Day Vol', 'Last', 'Bid', 'Ask', 'CALL', 'Strike', 'PUT', 'Bid', 'Ask', 'Last','Day Vol', 'Open Interest', 'Settl.'])
    result = result.drop(0)
    result = result.drop(1)
    result = result.reset_index(drop=True)
    
    
    
    colmn1 = result['PUT'].tolist()
    colmn2 = result['CALL'].tolist()  
        
    for i in range(0,len(result)):
    
        if(colmn1[i] == "P" ):
            colmn1[i] = "PUT"
        if(colmn2[i] == "C" ):
            colmn2[i] = "CALL"
    
    a1 = pd.DataFrame({'PUT': colmn1})
    result['PUT'] = a1
    a2 = pd.DataFrame({'CALL': colmn2})
    result['CALL'] = a2
    
    result.columns = ['Settl.', 'Open Interest', 'Day Vol', 'Last', 'Bid', 'Ask', 'Option Type', 'Strike', 'Option Type', 'Bid', 'Ask', 'Last','Day Vol', 'Open Interest', 'Settl.']
    df1 = result.iloc[:, :8]
    df2 = result.iloc[:, 7:]
    
    new_data_frame = pd.concat([df1, df2], ignore_index=True)
    
    colmn = new_data_frame['Open Interest'].tolist()
    for i in range(0,len(new_data_frame)):
        if(colmn[i] == "-"):
            colmn[i] = 0
    a3 = pd.DataFrame({'Open Interest': colmn})
    new_data_frame['Open Interest'] = a3
        
    
    folder_path = "C:\\Users\\Admin\\Desktop\\Week_2\\Options_before_Euronext\\" 
    new_data_frame.to_csv(folder_path + url_togo[k][1] + today +'.csv')
    










