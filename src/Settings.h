/*
 * Settings.h
 *
 *  Created on: Jan 5, 2012
 *      Author: CK / João Carlos Ferreira Gonçalves
 */

#ifndef SETTINGS_H_
#define SETTINGS_H_

#include <vector>
using namespace std;

class Argument {

public:
	char* argument;
	// this could be updated (in the future) to have multiple values
	char* value;

	Argument() {
		argument = NULL;
		value = NULL;
	}
};

class Settings {

private:
	vector<Argument*> *argument_list;
	int argc;
	char **argv;

	void createSettings() {
		//first argument is the executable, so ignore it
		int argc_m1 = argc - 1;
		for (int i = 1; i < argc; i++) {
			//	cout << argv[i] << endl;
			//arguments must be on the form
			//-arg <val>
			//where val can be non-existent
			char* cur_arg = argv[i];
			//if it is an argument, it begins with a "-"
			if (cur_arg[0] == '-') {
				Argument *a = new Argument();
				a->argument = cur_arg;
				argument_list->push_back(a);
				//check if next string is the value
				if (i < argc_m1) { //obviously if there are more arguments
					char* next_arg = argv[i + 1];
					if (next_arg[0] != '-') {
						a->value = next_arg;
						i++;
					}
				}
			}
			//		cout << argument_list->size() << endl;
		}
	}

public:
	/**
	 * receives the same arguments as the main function
	 */
	Settings(int argc, char **argv) {
		argument_list = new vector<Argument*>();
		this->argc = argc;
		this->argv = argv;
		this->createSettings();
	}

	~Settings() {
		for (unsigned int i = 0; i < argument_list->size(); i++) {
			delete argument_list->at(i);
		}
		delete argument_list;
	}

	Argument* getArgument(int pos) {
		return this->argument_list->at(pos);
	}

	int getAmountArguments() {
		return this->argument_list->size();
	}
};

#endif /* SETTINGS_H_ */
