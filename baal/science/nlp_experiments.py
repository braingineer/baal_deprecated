from baal import utils, crf, nlp
import pickle
import random

def nps_crf_data_create(splits=(0.6,0.2,0.2)):
    session_generator = nlp.nps_chat_interface.nps.make_sessions()
    sessions = {session.name: session for session in session_generator}

    all_data = {}
    all_test = {}

    for session in sessions.values():
        session.filter_system_posts()
        raw_data = list(session.pos_data_genesis())
        random.shuffle(raw_data)
        print "\nRaw Data length: %d" % len(raw_data)
        n1,n2 = int(len(raw_data)*splits[0]), int(len(raw_data)*(splits[0]+splits[1]))
        print "Splitting. Up to %d is training" % n1
        print "From %d to %d for dev" % (n1, n2)
        print "From %d to %d for test" % (n2, len(raw_data))
        training = raw_data[:n1]
        dev = raw_data[n1:n2]
        test = raw_data[n2:]
        print "trust but verify: %d, %d, %d" % (len(training), len(dev), len(test))
        all_data[session.name] = (training,dev)
        all_test[session.name] = test

    with open("nps_data.pkl", 'wb') as fp:
        pickle.dump(all_data, fp)

    with open("nps_test.pkl", 'wb') as fp:
        pickle.dump(all_test, fp)


def nps_crf_make(vaultable=True):
    """
    This will run a demo of the CRF on the NPS Chat dataset
    Returns:
            nps_crf,check_chain

            nps_crf is the crf object
            check_chain is a function that displays actual vs predicted
    """
    if utils.vault.exists('crf_debug_v3') and vaultable:
        print "Found a version in the vault. Getting it."
        return utils.vault.retrieve('crf_debug_v3', crf.linear_crf.LinearCRF)

    session_generator = nlp.nps_chat_interface.nps.make_sessions()
    sessions = {session.name: session for session in session_generator}
    print "Available Sessions:\n\t%s\n" % "\n\t".join(sessions.keys())

    print "Demonstration at random"
    demonstration_session = sessions.values()[random.randint(0,
                                                          (len(sessions) - 1))]
    print "Chose %s" % demonstration_session.name

    demonstration_session.filter_system_posts()

    print "First 10 posts in session: %s" % "\n\t".join([str(c) for c in demonstration_session[:10]])

    raw_data = demonstration_session.pos_data_genesis()

    print "Making the CRF proper"

    nps_crf = crf.linear_crf.LinearCRF.make(raw_data)

    if utils.vault.store('crf_debug_v3', nps_crf,
                         "Testing and Validating CRF Functionality"):
        print "Successfully vaulted."
    else:
        print "Failed to vault"

    return nps_crf

def nps_crf_test(nps_crf=None):
    print "Running the learner!"

    if nps_crf is None:
        if utils.vault.exists('crf_debug'):
            nps_crf = utils.vault.retrieve('crf_debug_v3', crf.linear_crf.LinearCRF)
        else:
            raise utils.AlgorithmicException("Feed me CRFs. Signed, [nps_crf_test]")

    nps_crf.run(nps_crf.l_bfgs)

    def check_chain(i, dev=True):
        if dev:
            chains = nps_crf.dev
        else:
            chains = nps_crf.test

        if i >= len(chains):
            print "Pass back a smaller i. Defaulting to 0"
            i = 0
        chain = chains[i]

        best_path = nps_crf.viterbi(chain)
        best_labels = nps_crf.label_vocab.lookup_many(best_path)
        comparing = ["%s: %s vs %s" % (word, actual, predicted) for
                     word, actual, predicted in zip(chain.tokens, chain.true_labels, best_labels)]

        print "Word: Actual vs Predicted:\n\t %s" % "\n\t".join(comparing)
    return nps_crf, check_chain
