import gzip
import json


file = '/Users/cipherskin/PycharmProjects/proverb/data/manually_added/'

if '6_Minute_Walk_Test' in str(file):
    pass
else:
    print('Not A 6MWT')

try:
    # Fetch the file from locally stored directory
    with gzip.open(file) as fin:
        data = fin.read()
        data = json.loads(data.decode('utf-8'))

    if 'sit_to_stand_count' in data['rawData'].keys():
        final_distance = data['rawData']['sit_to_stand_count']
        print(f'Distance: {final_distance} m')

except:
    print('Error')
