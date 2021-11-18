
import web_driver as WD

def main():
    driver = WD.get_driver()
    driver.get()


def get_gmarket_data():
    f = open('products.txt', 'r')
    products = f.readlines()

    driver = WD.get_driver(False)



    try:
        for url in products:
            url = url.strip()
            driver.get(url)
    except Exception as e:
        print(e)
    finally:
        driver.quit()
    return

get_gmarket_data()