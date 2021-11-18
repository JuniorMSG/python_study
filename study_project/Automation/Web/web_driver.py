from selenium import webdriver
from selenium.common.exceptions import TimeoutException
import random
import re
import chromedriver_autoinstaller

def driver_set(open_browser):

    options = webdriver.ChromeOptions()

    if open_browser:
        options.add_argument('headless')
        # options.add_argument('--headless')

    options.add_argument("debuggerAddress=127.0.0.1:9222")
    options.add_argument('window-size=1920x1080')
    options.add_argument("--disable-gpu")
    options.add_argument("lang=ko_KR")

    # options.add_argument('disable-infobars')
    options.add_argument('start-maximized')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False
    }
    options.add_experimental_option("prefs", prefs)

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
    options.add_argument('user-agent=' + user_agent)

    # firefox_profile = webdriver.FirefoxProfile()
    # firefox_profile.set_preference('general.useragent.override',
    #                                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0')

    chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]  # 크롬드라이버 버전 확인

    try:
        driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe', options=options)
    except:
        chromedriver_autoinstaller.install(True)
        driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe', options=options)

    driver.create_options()
    driver.set_page_load_timeout(50)
    driver.set_window_size(1920, 1080)
    driver.maximize_window()



    return driver


# 유저 에이전트 환경 설정
def get_random_ua():

    all_user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; KTXN)",
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko",

        "Mozilla/4.0 (Windows; U; Windows NT 5.0; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.195.33 Safari/532.0",
        "Mozilla/4.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/525.19 (KHTML, like Gecko) Chrome/1.0.154.59 Safari/525.19",

        "Mozilla/4.0 (compatible; MSIE 6.0; Linux i686 ; en) Opera 9.70",
    ]
    random_ua_index = random.randint(0, len(all_user_agents) - 1)
    ua = re.sub(r"(\s)$", "", all_user_agents[random_ua_index])
    return ua


# 크롬드라이버 환경 설정
def get_driver(open_browser):

    options = webdriver.ChromeOptions()


    # if open_browser:
    #     options.add_argument('headless')

    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-certificate-errors-spki-list')
    options.add_argument('acceptInsecureCerts')
    options.add_argument('--allow-insecure-localhost')
    options.add_argument('--able-popup-blocking')
    options.add_argument('--log-level=3')
    options.add_argument("disable-gpu")
    options.add_argument("user-agent="+get_random_ua())
    options.add_argument("lang=ko_KR")


    # enable-automation 자동화 막대 비활성, load-extension 개발자 모드 확장 사용 비활성
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging", "load-extension" ])
    options.add_experimental_option('useAutomationExtension', False)

    # 로그인시 비밀번호창 비활성
    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False
    }
    options.add_experimental_option("prefs", prefs)

    chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]  # 크롬드라이버 버전 확인
    try:
        driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe', options=options)
    except:
        chromedriver_autoinstaller.install(True)
        driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe', options=options)

    driver.set_page_load_timeout(30)
    driver.create_options()
    driver.maximize_window()

    return driver

def browser_timeout(current_url, driver, FlagOp):
    error_cnt = 0
    import time
    import random



    while error_cnt < 5:
        try:
            if error_cnt == 0:

                driver.get(current_url)
                if FlagOp:
                    time.sleep(random.uniform(15, 30))
            else:
                driver.refresh()
                driver.switch_to.window(driver.window_handles[-1])

            error_cnt = 5
            return True
        except TimeoutException as e:
            print(e)
            error_cnt += 1
            if error_cnt == 5:
                print("타임아웃 에러 5회이상")
                return False


