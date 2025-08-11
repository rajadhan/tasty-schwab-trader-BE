# Script to run the update_tastytrade_instruments_csv function

from tastytrade import update_tastytrade_instruments_csv

if __name__ == "__main__":
    print("Starting Tastytrade instruments update...")
    update_tastytrade_instruments_csv()
    print("Update complete!")