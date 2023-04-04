from helper import FuseBot

if __name__ == '__main__':
    bot = FuseBot()
    query_response_count = 0
    hard_skill_response_count = 0
    soft_skill_response_count = 0
    hr_message_count = 0
    goodbye_response_count = 0
    
    while query_response_count < 2:  # loop until two query responses
        bot.query_response()
        query_response_count += 1
    
    while hard_skill_response_count < 4:  # loop until four hard skill responses
        bot.hard_skill_response()
        hard_skill_response_count += 1
        print('*'*50)
        print(hard_skill_response_count)
        print('*'*50)

    while soft_skill_response_count < 2:
        bot.soft_skill_response()
        soft_skill_response_count += 1
    
    while hr_message_count < 3:
        bot.hr_message_reponse('index.json')
        hr_message_count += 1
    
    while goodbye_response_count < 1:
        bot.good_bye_response()
        goodbye_response_count += 1