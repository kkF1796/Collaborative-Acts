import string

import pandas as pd
import numpy as np

"""
import spacy
from spacy import displacy
spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_sm")"""


RIGHT_CORRECTED = [ "m'", "c'", "d'", "j'", "qu'", "l'",'commençent', "t'",'tattend','vis-à-vis',"r'prend",'faudrais','-ce',"s'",'persuadif','responsabilisant',"problème\'",'ttend', 'correpondante','est-',"n'",'vice-versa','ptite','ltourner', 'laime','celui-ci','peut-être','anglicisme','-là','au-dessus','entr','euh\'','problématiques','vacances/','convainre','jeuns','elle-même','être/','bien-être','eb','impliqué/',"jusqu'",'bohneur','lenlève','hm¨','baîllement','quelques-uns','camrade','buzzé','buzze','-moi','insideuse','societé', '-vous', 'lui-même','anesthesié','mots-clés','recollaboré','sélectionnant', 'clicks' ,'contre-productif','collégiale','.)pff','ré-écrire','chock','lannule','faudrati','metttre','réfléxion','ex-aequo',"puisqu'",'comettent', "aujoud'" ,'integrité',"quelqu'",'exatement','rebéllion','originalit','récuperées','lutilise','cylce', 'mettres','positi','adaptabilité', 'occurence',"jeun'",'vendale','pouruqoi','préfèrais','préferé','hesité','consideré','dénonc','préferez','abrutit','accroit','rouvres',
'empèche','linguistiquement','pesuasif','réfléchitr','harcèllement','accroch','pluseurs','paraîssait','tutoiement','impértatif','élement','neurons','rappport','violen','asapté','epunition','pourraît','peutx','enragee','demolir','lsogans','enricihir','etceatera','reflexion','arretons','adol','.)oui', 'mechants','.)mais','acilement','triatements','experimenté',"q'",'préferés','preferés','foutrre','medre','provocatine','-en','élements','réflchir','caïb','drecte','prète','.)moi'] 

LIST_1= ['jpensais', 'ptetre',  'ptretre',  'dja', 'tsais','mamuse', 'vectrice', 'shif' , '2ème', 'vocab', 'v(inaud', 'têtre', 'forcém', '3ème',  'vacréer',  'trams', 'clické', 'accus',  'persu', 'leouais', 'clicker',  'interroga', 'clicke', '1er', 'estplus', '5min','adalabilité', 'scuse', 'rééexplique', 'exprime.toi', 'preferé', 'tempis', 'preferes','3h', 'ambigü', 'longuet', 'tentends', 'cêst', ".)c'", 'c0est', "ouaisj'", 'etpuis', 'langage+smiley', '4ème', 'clickes', 'accord?e', 'rire)mais', 'o?u', 'onva', 'caid', 'clickez', 'traument', 'exclam', 'persua','persuas', 'émot', 'guid', 'chenis', 'slo', 'genrene', 'quizaine',  'adapt',  "moij'", 'cli', 'succ', 'mobyinm', 'tebé', 'scue', 'paren', 'estse', 'utilisos', 'propore', 'cérer','leut', 'avoie', 'consis', 'onf', 'rassurance', 'appelatif', 'forcep', 'brouté', 'contine','persua', 'tieng','êre', 'impr', 'evoaue','anour', 'aurte', 'rire)r', 'frappet', 'gaches', 'perdents', 'vei', 'ièche','orig', 'doite', "j''met", 'vocabulairement', 'sogan', 'rapeur', 'raper','flè', 'oup', 'tsais', 'matu']

CORRECT_1 = [['je','pensais'], ['peut-être'], ['peut-être'], ['déjà'], ['tu', 'sais'], ["m'",'amuse'], ['vecteur'], ['shift'], ['deuxième'], ['vocabulaire'], ['inaud'], ['peut-être'], ['forcément'], ['troisième'], ['va','créer'], ['tramways'], ['cliqué'], ['accord'], ['perçu'], ['ouais'], ['cliquer'], ['interrogation'], ['clique'],['premier'], ['est','plus'], ['cinq','minutes'], ['adptabilité'], ['excuse'], ['réexplique'], ['exprime','toi'], ['préféré'], ['tant','pis'], ['préfères'], ['trois','heures'], ['ambigu'], ['long'], ['tu','entends'], ["c'",'est'], ["c'"], ["c'",'est'], ['ouais','je'], ['et', 'puis'], ['language','+','smilley'], ['quatrième'], ['cliques'], ['accord', '?'], ['rire','mais'], ['ou'], ['on','va'], ['caïd'], ['cliquer'], ['autrement'], ['exclamation'], ['persuasif'],['persuasif'],['émotion'], ['guide'], ['bazar'], ['slogan'],['genre','ne'],['dizaine'], ['adapté'],['moi','je'], ['clics'],['succès'],['mobying'], ['débile'], ['excuse'], ['parenthèse'],['est','se'], ['utilisons'], ['propre'],['créer'], ['leur'],['avoir'],['concis'],['on'],['rassurement'],['appelant'],['force'],['trompé'],['continue'],['persuade'],['tien'],['être'],['impression'],['et','ouais'],['amour'],['autre'],['rire'], ['frapper'], ['gâches'],['perdent'],['vie'],['chier'],['original'],['droite'],['je','mets'],['avec','les','mots'], ['slogan'],['rappeur'],['rapper'],['flèche'],['oups'],['tu','sais'],['diplôme']]

DELETE =['sug', 'ife', 'fdis', 'cama','ff', 'dél', 'prin', 'frac', 'surt', 'pisé', 'obje', 'meu','moré',  'fiut',  'murger', 'teur', 'fau', 'entrai', 'enchi',  'botch', 'éle', 'répé','phr',  'aec', 'exc', 'corr']

PONCT=['...', '..',  '.....']

# NEED TO DECIDE WHAT TO DO WITH THESE WORDS
NOMS_PROPRES=['benetton', 'plainpalais','lancy', 'colombine', 'titeuf']
NAME=['aurélie', 'stael', 'frédirique', 'yohan', 'floriant']

VOCAB_JEUN_1=['teubé', 'boloss',  'zyva', 'stoss', 'stos', 'swap', 'mifa', 'keums', 'péta', 'rangnangnan']
VOCAB_JEUN_2=['branchouillé', "adol'", 'adolechiant', 'adole', 'attachiant', 'addolechiante', 'attachiante', 'adolechiante', 'supprimike', 'adolaimant',]
JEUN_CORRECT_2=[['branché'],['adolescent'], ['adolescent', 'chiant'], ['adolescent'], ['attachant','chiant'], ['adolescent', 'chiante'],['attachant','chiant'],['adolescent', 'chiante'], ['supprimer'],['adolescent','aimant']]

INTERJ_del = [ 'touc', 'tuc', 'toing',  'hihi','poupoupou',  'youp', 'alala', 'boh', 'tatata',  'gnagna', 'nanani','oulala', 'olala' ] # 'punkt' , 'lalala', 'ouuuups' # à partir de tatata/ gnagna/nanani ~ blablabla - oulala/olala/ouuuups

UNKNOWN=[ "p'", 'vio','génèreg',  'ctif', 'lence', 'représ', 'précuit', 'déterrant', 'jardon', 'violà', 'instiller', 'quarty', 'caps', 'niquait', 'gera', 'rega', 'rainte', 'tage']# à verifier


def correct_jWORDS(tokens):
	toks = []
	for token in tokens:
		if token != '':
			if token[0]=='j' and len(token) !=1 and token[1].isalpha() and not token[1] in "aeiou":
				toks.append('je')
				t = token[1:len(token)+1]
				toks.append(t) # ou ajouter LEMMA
			else:
				toks.append(token)
	return toks
