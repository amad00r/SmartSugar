class Schedule:
    def __init__(self):
        self.OPT_TIME_SPACE = 3

        self.OPT_BREAKFAST = None
        self.OPT_SECOND_BREAKFAST = None
        self.OPT_LAUNCH = None
        self.OPT_AFTERNOON_SNACK = None
        self.OPT_DINNER = None

        self.first_meal_done = False

    def first_meal(self, moment):
        try:
            self.OPT_BREAKFAST = float(moment)
            if self.OPT_BREAKFAST + 6 >= 16: 
                self.OPT_SECOND_BREAKFAST = None
                self.OPT_LAUNCH = self.OPT_BREAKFAST + self.OPT_TIME_SPACE
                self.OPT_AFTERNOON_SNACK = self.OPT_BREAKFAST + self.OPT_TIME_SPACE * 2
                self.OPT_DINNER = self.OPT_BREAKFAST + self.OPT_TIME_SPACE * 3
            else:
                self.OPT_SECOND_BREAKFAST = self.OPT_BREAKFAST + self.OPT_TIME_SPACE
                self.OPT_LAUNCH = self.OPT_BREAKFAST + self.OPT_TIME_SPACE * 2
                self.OPT_AFTERNOON_SNACK = self.OPT_BREAKFAST + self.OPT_TIME_SPACE * 3
                self.OPT_DINNER = self.OPT_BREAKFAST + self.OPT_TIME_SPACE * 4
            self.first_meal_done = True
        except:
            print('''Error. Introdueix l'hora del primer àpat amb les hores en format 24h i els minuts en format decimal. Ex: 09:30h = 9.5''')
            self.first_meal_done = False

    def print_all(self):
        if self.first_meal_done == True:
            print('Esmorzar: ' + str(self.OPT_BREAKFAST))
            print('Segon esmorzar: ' + str(self.OPT_SECOND_BREAKFAST))
            print('Dinar: ' + str(self.OPT_LAUNCH))
            print('Berenar: ' + str(self.OPT_AFTERNOON_SNACK))
            print('Sopar: ' + str(self.OPT_DINNER))
        else:
            print('Encara no has introduït el primer àpat')

    