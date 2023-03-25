from policies.RandomPolicy import RandomTTTPolicy, main

if __name__ == "__main__":
    random_policy = RandomTTTPolicy()
    random_policy.select_action()
    main()
