import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import sklearn
import pandas as pd

# init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


# database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + \
    os.path.join(basedir, 'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# init db
db = SQLAlchemy(app)

# init marshmallow
ma = Marshmallow(app)

# training status
global status
global classifier
status = 0
classifier = None

#classification model
def training(test):
    print("doing training")
    url = "https://raw.githubusercontent.com/soun059/medical_recommend/master/dataset.csv"
    dataset = pd.read_csv(url)
    dataset = dataset.dropna()
    inputs = pd.get_dummies(dataset.Target)
    dataset_pre = pd.concat([dataset['Source'], inputs], axis=1)
    dataset_pre.drop_duplicates(inplace=True, keep='first')
    dataset_pre = dataset_pre.groupby('Source').sum().reset_index()
    y = dataset_pre['Source']
    x = dataset_pre.drop(['Source'], axis=1)
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(random_state=42)
    # from sklearn.model_selection import train_test_split
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # print(test)
    classifier.fit(x, y)
    y_pred = classifier.predict([test])
    # print(y_pred)
    # print(classifier.score(x_test, y_test))
    return y_pred
#########################

# User class
class User(db.Model):
    _id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    password = db.Column(db.String(100))
    age = db.Column(db.Integer)
    email = db.Column(db.String(200))

    def __init__(self, name, password, age, email):
        self.name = name
        self.password = password
        self.age = age
        self.email = email


# review class
class Review(db.Model):
    _id = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Integer)
    description = db.Column(db.String(200))
    user_id = db.Column(db.Integer)

    def __init__(self, rating, description, user_id):
        self.rating = rating
        self.description = description
        self.user_id = user_id


# review schema
class ReviewSchema(ma.Schema):
    class Meta:
        fields = ('_id', 'rating', 'description', 'user_id')

# User schema


class UserSchema(ma.Schema):
    class Meta:
        fields = ('_id', 'name', 'status')

# user return status


class UserReturnStatus(ma.Schema):
    class Meta:
        fields = ('status', 'name', 'id')


# symptomps
class symptomps(ma.Schema):
    class Meta:
        fields = ('status', 'symp_arr')


# disease
class disease_recommended(ma.Schema):
    class Meta:
        fields = ('status', 'disease')


review_schema = ReviewSchema()
reviews_schema = ReviewSchema(many=True)
user_schema = UserSchema()
users_schema = UserSchema(many=True)
user_return = UserReturnStatus()
symptomps_ = symptomps()
disease_recommended_ = disease_recommended()

# Create a Product
@app.route('/userrev', methods=['POST'])
def add_review():
    rating = request.json['rating']
    description = request.json['description']
    user_id = request.json['user_id']
    new_review = Review(rating, description, user_id)
    db.session.add(new_review)
    db.session.commit()
    return user_return.jsonify({
        "id": user_id,
        "status": "true",
    })


@app.route('/userlog', methods=['POST'])
def validator():
    email = request.json['email']
    password = request.json['password']
    email_val = User.query.filter_by(email=email).first()
    # pass_val = User.query.get(password)
    if(email_val):
        if(email_val.password == password):
            return user_schema.jsonify({'status': True, 'name': email_val.name, "_id": email_val._id})
        else:
            return user_schema.jsonify({'status': False, 'name': '', "_id": ''})
    else:
        return user_schema.jsonify({'status': False, 'name': '', "_id": ''})


# adding user
@app.route('/useradd', methods=['POST'])
def add_user():
    name = request.json['name']
    password = request.json['password']
    email = request.json['email']
    age = request.json['age']
    new_user = User(name, password, age, email)
    db.session.add(new_user)
    db.session.commit()
    return user_return.jsonify({
        "id": new_user._id,
        "status": "true"
    })


# symptomps caller
@app.route('/symptomps', methods=['GET'])
def get_symptomps():
    symp_arr = ['feeling suicidal', 'suicidal', 'hallucinations auditory', 'feeling hopeless', 'weepiness', 'sleeplessness', 'motor retardation', 'irritable mood', 'blackout', 'mood depressed', 'hallucinations visual', 'worry', 'agitation', 'tremor', 'intoxication', 'verbal auditory hallucinations', 'energy increased', 'difficulty', 'nightmare', 'unable to concentrate', 'homelessness', 'thicken', 'tumor cell invasion', 'metastatic lesion', 'pain neck', 'lung nodule', 'pain', 'pain abdominal', 'food intolerance', 'mass of body structure', 'atypia', 'shortness of breath', 'prostatism', 'drowsiness', 'sleepy', 'hyponatremia', 'fall', 'unsteady gait', 'polyuria', 'hypotension', 'syncope', 'speech slurred', 'weight gain', 'asthenia', 'fatigue', 'tired', 'mental status changes', 'vomiting', 'numbness', 'hematuria', 'dysesthesia', 'polymyalgia', 'passed stones', 'pleuritic pain', 'guaiac positive', 'rale', 'breath sounds decreased', 'urge incontinence', 'dysuria', 'diarrhea', 'vertigo', 'qt interval prolonged', 'ataxia', 'paresis', 'hemianopsia homonymous', 'hemodynamically stable', 'rhonchus', 'orthostasis', 'decreased body weight', 'chill', 'monoclonal', 'ecchymosis', 'haemorrhage', 'pallor', 'heme positive', 'pain back', 'dizziness', 'arthralgia', 'swelling', 'transaminitis', 'nausea', 'cushingoid facies', 'cushingoid habitus', 'ascites', 'sore to touch', 'apyrexial', 'palpitation', 'splenomegaly', 'pruritus', 'distended abdomen', 'lesion', 'bleeding of vagina', 'dyspnea', 'cardiovascular finding', 'cardiovascular event', 'haemoptysis', 'cough', 'hypercapnia', 'soft tissue swelling', 'ambidexterity', 'fever', 'stool color yellow', 'rigor - temperature-associated observation', 'night sweat', 'spontaneous rupture of membranes', 'patient non compliance', 'tachypnea', 'productive cough', 'muscle hypotonia', 'hypotonic', 'has religious belief', 'disturbed family', 'behavior hyperactive', 'catatonia', 'hypersomnia', 'hyperhidrosis disorder', 'mydriasis', 'extrapyramidal sign', 'loose associations', 'exhaustion', 'unresponsiveness', 'hypothermia natural', 'incoherent', 'lameness', 'claudication', 'unconscious state', 'clammy skin', 'distress respiratory', 'ache', 'macerated skin', 'heavy feeling', 'asterixis', 'constipation', 'general discomfort', 'urinary hesitation', 'dizzy spells', 'shooting pain', 'bradycardia', 'systolic ejection murmur', 'hyperemesis', 'polydypsia', 'paresthesia', 'titubation', 'dysarthria', 'painful swallowing', 'hoarseness', 'stridor', 'spasm', 'dysdiadochokinesia', 'achalasia', 'stiffness', 'side pain', 'emphysematous change', 'welt', 'tinnitus', 'hydropneumothorax', 'superimposition', 'difficulty passing urine', 'seizure', 'enuresis', 'lethargy', 'consciousness clear', 'muscle twitch', 'headache', 'lightheadedness', 'out of breath', 'sedentary', 'angina pectoris', 'unhappy', 'labored breathing', 'hematocrit decreased', 'wheezing', 'hypoxemia', 'renal angle tenderness', 'feels hot/feverish', 'general unsteadiness', 'facial paresis', 'hemiplegia', 'dyspnea on exertion', 'asymptomatic', 'hypokinesia', 'left atrial hypertrophy', 'cardiomegaly', 'chest discomfort', 'urgency of micturition', 'orthopnea', "Heberden's node", 'jugular venous distention', 'sweat', 'sweating increased', 'hyperkalemia', 'sinus rhythm', 'pain chest', 'feeling strange', 'pustule', 'estrogen use', 'hypometabolism', 'aura', 'myoclonus', 'gurgle', 'wheelchair bound', 'yellow sputum', 'cachexia', 'myalgia',
                'neck stiffness', 'hacking cough', 'dyspareunia', 'hypokalemia', 'poor dentition', 'non-productive cough', 'floppy', 'mediastinal shift', 'clonus', 'decreased translucency', 'extreme exhaustion', 'stupor', 'pressure chest', 'chest tightness', 'nausea and vomiting', 'awakening early', 'fatigability', 'tenesmus', 'slowing of urinary stream', 'monocytosis', 'posterior rhinorrhea', 'fremitus', 'decreased stool caliber', 'satiety early', 'hematochezia', 'egophony', 'cicatrisation', 'scar tissue', 'paraparesis', 'moody', 'fear of falling', 'breech presentation', 'cyanosis', 'abscess bacterial', 'abdomen acute', 'air fluid level', 'catching breath', 'abdominal tenderness', 'flatulence', 'gravida 0', 'throat sore', 'hepatosplenomegaly', 'snuffle', 'hoard', 'neologism', 'panic', 'lip smacking', 'phonophobia', 'rolling of eyes', 'hirsutism', 'absences finding', 'fecaluria', 'projectile vomiting', 'pneumatouria', 'cystic lesion', 'anorexia', 'hunger', 'nervousness', 'aphagia', 'focal seizures', 'abnormal sensation', "Stahli's line", 'stinging sensation', 'paralyse', 'hot flush', 'redness', 'erythema', 'hypocalcemia result', 'oliguria', 'rhd positive', 'heartburn', 'heavy legs', 'drool', 'pin-point pupils', 'bedridden', 'frail', 'tremor resting', 'groggy', 'impaired cognition', 'macule', 'photophobia', 'scratch marks', 'sniffle', 'numbness of hand', 'bradykinesia', 'unwell', 'sensory discomfort', 'history of - blackout', 'hyperacusis', 'hepatomegaly', 'breakthrough pain', 'green sputum', 'hypoproteinemia', 'colic abdominal', 'hypertonicity', 'hypoalbuminemia', 'hypersomnolence', 'underweight', 'withdraw', 'terrify', 'decompensation', 'uncoordination', 'posturing', 'tonic seizures', 'debilitation', 'pain in lower limb', 'transsexual', 'nonsmoker', 'prostate tender', 'pain foot', 'mass in breast', 'snore', 'bruit', 'disequilibrium', 'bowel sounds decreased', 'burning sensation', 'verbally abusive behavior', 'adverse reaction', 'adverse effect', 'abdominal bloating', 'no status change', 'pansystolic murmur', 'room spinning', 'indifferent mood', 'st segment depression', 't wave inverted', 'giddy mood', 'homicidal thoughts', 'pulsus paradoxus', 'gravida 10', 'dullness', 'milky', 'regurgitates after swallowing', 'vision blurred', 'systolic murmur', 'sciatica', 'frothy sputum', 'rest pain', 'large-for-dates fetus', 'para 1', 'immobile', 'malaise', 'moan', "Murphy's sign", 'gasping for breath', 'feces in rectum', 'abnormally hard consistency', 'stuffy nose', 'presence of q wave', 'photopsia', 'barking cough', 'rapid shallow breathing', 'noisy respiration', 'nasal discharge present', 'symptom aggravating factors', 'retropulsion', 'formication', 'hypesthesia', 'sputum purulent', 'low back pain', 'charleyhorse', 'pericardial friction rub', 'scleral icterus', 'nasal flaring', 'sneeze', 'prodrome', 'rambling speech', 'clumsiness', 'flushing', 'urinoma', 'throbbing sensation quality', 'hyperventilation', 'excruciating pain', 'gag', 'pulse absent', 'flare', 'st segment elevation', 'anosmia', 'para 2', 'abortion', 'intermenstrual heavy bleeding', 'previous pregnancies 2', 'primigravida', 'proteinemia', 'breath-holding spell', 'retch', 'no known drug allergies', 'inappropriate affect', 'poor feeding', 'todd paralysis', 'alcoholic withdrawal symptoms', 'red blotches', 'behavior showing increased motor activity', 'coordination abnormal', 'choke', 'alcohol binge episode', 'blanch', 'elation', 'r wave feature', 'overweight']
    return symptomps_.jsonify({
        'status': True,
        'symp_arr': symp_arr
    })

# disease predictor
@app.route('/recommend', methods=['POST'])
def get_recommend():
    global status
    global classifier
    symp_arr = ['Heberden\'s node', 'Murphy\'s sign', 'Stahli\'s line', 'abdomen acute', 'abdominal bloating', 'abdominal tenderness', 'abnormal sensation', 'abnormally hard consistency', 'abortion', 'abscess bacterial', 'absences finding', 'achalasia', 'ache', 'adverse effect', 'adverse reaction', 'agitation', 'air fluid level', 'alcohol binge episode', 'alcoholic withdrawal symptoms', 'ambidexterity', 'angina pectoris', 'anorexia', 'anosmia', 'aphagia', 'apyrexial', 'arthralgia', 'ascites', 'asterixis', 'asthenia', 'asymptomatic', 'ataxia', 'atypia', 'aura', 'awakening early', 'barking cough', 'bedridden', 'behavior hyperactive', 'behavior showing increased motor activity', 'blackout', 'blanch', 'bleeding of vagina', 'bowel sounds decreased', 'bradycardia', 'bradykinesia', 'breakthrough pain', 'breath sounds decreased', 'breath-holding spell', 'breech presentation', 'bruit', 'burning sensation', 'cachexia', 'cardiomegaly', 'cardiovascular event', 'cardiovascular finding', 'catatonia', 'catching breath', 'charleyhorse', 'chest discomfort', 'chest tightness', 'chill', 'choke', 'cicatrisation', 'clammy skin', 'claudication', 'clonus', 'clumsiness', 'colic abdominal', 'consciousness clear', 'constipation', 'coordination abnormal', 'cough', 'cushingoid facies', 'cushingoid habitus', 'cyanosis', 'cystic lesion', 'debilitation', 'decompensation', 'decreased body weight', 'decreased stool caliber', 'decreased translucency', 'diarrhea', 'difficulty', 'difficulty passing urine', 'disequilibrium', 'distended abdomen', 'distress respiratory', 'disturbed family', 'dizziness', 'dizzy spells', 'drool', 'drowsiness', 'dullness', 'dysarthria', 'dysdiadochokinesia', 'dysesthesia', 'dyspareunia', 'dyspnea', 'dyspnea on exertion', 'dysuria', 'ecchymosis', 'egophony', 'elation', 'emphysematous change', 'energy increased', 'enuresis', 'erythema', 'estrogen use', 'excruciating pain', 'exhaustion', 'extrapyramidal sign', 'extreme exhaustion', 'facial paresis', 'fall', 'fatigability', 'fatigue', 'fear of falling', 'fecaluria', 'feces in rectum', 'feeling hopeless', 'feeling strange', 'feeling suicidal', 'feels hot/feverish', 'fever', 'flare', 'flatulence', 'floppy', 'flushing', 'focal seizures', 'food intolerance', 'formication', 'frail', 'fremitus', 'frothy sputum', 'gag', 'gasping for breath', 'general discomfort', 'general unsteadiness', 'giddy mood', 'gravida 0', 'gravida 10', 'green sputum', 'groggy', 'guaiac positive', 'gurgle', 'hacking cough', 'haemoptysis', 'haemorrhage', 'hallucinations auditory', 'hallucinations visual', 'has religious belief', 'headache', 'heartburn', 'heavy feeling', 'heavy legs', 'hematochezia', 'hematocrit decreased', 'hematuria', 'heme positive', 'hemianopsia homonymous', 'hemiplegia', 'hemodynamically stable', 'hepatomegaly', 'hepatosplenomegaly', 'hirsutism', 'history of - blackout', 'hoard', 'hoarseness', 'homelessness', 'homicidal thoughts', 'hot flush', 'hunger', 'hydropneumothorax', 'hyperacusis', 'hypercapnia', 'hyperemesis', 'hyperhidrosis disorder', 'hyperkalemia', 'hypersomnia', 'hypersomnolence', 'hypertonicity', 'hyperventilation', 'hypesthesia', 'hypoalbuminemia', 'hypocalcemia result', 'hypokalemia', 'hypokinesia', 'hypometabolism', 'hyponatremia', 'hypoproteinemia', 'hypotension', 'hypothermia natural', 'hypotonic', 'hypoxemia', 'immobile', 'impaired cognition', 'inappropriate affect', 'incoherent', 'indifferent mood', 'intermenstrual heavy bleeding', 'intoxication',
                'irritable mood', 'jugular venous distention', 'labored breathing', 'lameness', 'large-for-dates fetus', 'left atrial hypertrophy', 'lesion', 'lethargy', 'lightheadedness', 'lip smacking', 'loose associations', 'low back pain', 'lung nodule', 'macerated skin', 'macule', 'malaise', 'mass in breast', 'mass of body structure', 'mediastinal shift', 'mental status changes', 'metastatic lesion', 'milky', 'moan', 'monoclonal', 'monocytosis', 'mood depressed', 'moody', 'motor retardation', 'muscle hypotonia', 'muscle twitch', 'myalgia', 'mydriasis', 'myoclonus', 'nasal discharge present', 'nasal flaring', 'nausea', 'nausea and vomiting', 'neck stiffness', 'neologism', 'nervousness', 'night sweat', 'nightmare', 'no known drug allergies', 'no status change', 'noisy respiration', 'non-productive cough', 'nonsmoker', 'numbness', 'numbness of hand', 'oliguria', 'orthopnea', 'orthostasis', 'out of breath', 'overweight', 'pain', 'pain abdominal', 'pain back', 'pain chest', 'pain foot', 'pain in lower limb', 'pain neck', 'painful swallowing', 'pallor', 'palpitation', 'panic', 'pansystolic murmur', 'para 1', 'para 2', 'paralyse', 'paraparesis', 'paresis', 'paresthesia', 'passed stones', 'patient non compliance', 'pericardial friction rub', 'phonophobia', 'photophobia', 'photopsia', 'pin-point pupils', 'pleuritic pain', 'pneumatouria', 'polydypsia', 'polymyalgia', 'polyuria', 'poor dentition', 'poor feeding', 'posterior rhinorrhea', 'posturing', 'presence of q wave', 'pressure chest', 'previous pregnancies 2', 'primigravida', 'prodrome', 'productive cough', 'projectile vomiting', 'prostate tender', 'prostatism', 'proteinemia', 'pruritus', 'pulse absent', 'pulsus paradoxus', 'pustule', 'qt interval prolonged', 'r wave feature', 'rale', 'rambling speech', 'rapid shallow breathing', 'red blotches', 'redness', 'regurgitates after swallowing', 'renal angle tenderness', 'rest pain', 'retch', 'retropulsion', 'rhd positive', 'rhonchus', 'rigor - temperature-associated observation', 'rolling of eyes', 'room spinning', 'satiety early', 'scar tissue', 'sciatica', 'scleral icterus', 'scratch marks', 'sedentary', 'seizure', 'sensory discomfort', 'shooting pain', 'shortness of breath', 'side pain', 'sinus rhythm', 'sleeplessness', 'sleepy', 'slowing of urinary stream', 'sneeze', 'sniffle', 'snore', 'snuffle', 'soft tissue swelling', 'sore to touch', 'spasm', 'speech slurred', 'splenomegaly', 'spontaneous rupture of membranes', 'sputum purulent', 'st segment depression', 'st segment elevation', 'stiffness', 'stinging sensation', 'stool color yellow', 'stridor', 'stuffy nose', 'stupor', 'suicidal', 'superimposition', 'sweat', 'sweating increased', 'swelling', 'symptom aggravating factors', 'syncope', 'systolic ejection murmur', 'systolic murmur', 't wave inverted', 'tachypnea', 'tenesmus', 'terrify', 'thicken', 'throat sore', 'throbbing sensation quality', 'tinnitus', 'tired', 'titubation', 'todd paralysis', 'tonic seizures', 'transaminitis', 'transsexual', 'tremor', 'tremor resting', 'tumor cell invasion', 'unable to concentrate', 'unconscious state', 'uncoordination', 'underweight', 'unhappy', 'unresponsiveness', 'unsteady gait', 'unwell', 'urge incontinence', 'urgency of micturition', 'urinary hesitation', 'urinoma', 'verbal auditory hallucinations', 'verbally abusive behavior', 'vertigo', 'vision blurred', 'vomiting', 'weepiness', 'weight gain', 'welt', 'wheelchair bound', 'wheezing', 'withdraw', 'worry', 'yellow sputum']
    req_symp = request.json['req_symp']
    req_symp = req_symp[1:len(req_symp)-1].split(',')
    dataset = []
    for i in req_symp:
        dataset.append(i[1:len(i)-1])
    fd = []
    # print(dataset)
    for i in symp_arr:
        flag = False
        for j in dataset:
            # print(i, j)
            if i == j:
                fd.append(1)
                flag = True
        if flag == False:
            fd.append(0)
    print(fd)
    return disease_recommended_.jsonify({
        "status": True,
        "disease": training(fd)[0]  # p.iloc[0]
    })

    # Run Server
if __name__ == "__main__":
    app.run(debug=True)
