from baal import utils, crf, nlp
import random

def nps_crf():
    """
    This will run a demo of the CRF on the NPS Chat dataset
    Returns:
            nps_crf,check_chain

            nps_crf is the crf object
            check_chain is a function that displays actual vs predicted
    """
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

    nps_crf = crf.linear_crf.LinearCRF.make(raw_data)

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
