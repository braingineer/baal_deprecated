import json
import urllib2

class xkcd_json_worker:
    def __init__(self):
        self.blank_url = "http://xkcd.com/%s/info.0.json"
        self.current = None
        self.func_mid = None
        
    def __parse_transcript(self, json_data):
        transcript = [k.split("\n") for k in json_data['transcript'].split("\n\n")]
        return transcript
        
    def __call__(self, *args):
        self.current=json.load(urllib2.urlopen(self.blank_url % args[0]))
    
    def __get(self, arg):
        if self.current: return self.current[arg]
        else: raise NotImplemented
        
    def get(self, arg):
        if arg=='transcript': return self.__parse_transcript(self.current)
        else: return self.__get(arg)
        
    def pretty_print(self):
        for key in self.current.keys():
            if key!="transcript":  print key, "\t", self.current[key]
            else: print key, "\n", self.__parse_transcript(self.current)