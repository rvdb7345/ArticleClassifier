import glob
import numpy as np
import pandas as pd
from xml.dom import minidom
import xml.etree.ElementTree as ET


# -- Introduction --
# Parsing of XML CARs for different purposes, e.g. document-level classification.

# TODO: add a parameter to choose between subheadings and checktags

def parse_document_classification(path="../../data/"):
    """
        Function to parse the needed input and labels for
        document-level classification from all XML files.
    """

    # Load XMLs and extract PUIs
    files = sorted(glob.glob(path + "*.xml"))

    # Define XML namespaces
    prefix_map = {"ns0": "http://www.elsevier.com/xml/ani/ani",
                  "ns1": "http://www.elsevier.com/xml/ani/common",
                  "ce": "http://www.elsevier.com/xml/ani/common"}

    # Define fixed data structure
    columns = ["file_name", "pui", "title", "keywords", "abstract", \
               "abstract_2", "authors", "labels_m", "labels_a"]

    # Load classification labels into a list, it can be Check tags, Subheadings, etc.
    # Note: uncomment different sections to yield different set of labels.
    # - check tags
    # check_tags = pd.read_csv(path + "check_tags.csv", sep=";", header=None)
    # clf_labels = check_tags.iloc[:, 0].tolist()
    # columns += clf_labels

    # - subheadings
    clf_labels = pd.read_csv(path + "subheadings.csv", sep=";", header=None).iloc[:, 0].tolist()
    columns += clf_labels

    # - check tags groups (not for pipeline - used for experimentation)
    # check_tags = pd.read_csv(path + "check_tags.csv", sep=";", header=None)
    # clf_labels = check_tags.iloc[:, 0].tolist()
    # columns += ["organism studied", "controlled study", "animal study type ", "human study type (clinical work)",\
    #            "specific study types", "diagnostic test accuracy studies", "study procedures",\
    #            "age", "gender"]

    # Iterate through all files and documents
    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()

        data = []
        for item in root:

            # Get identifiers and input text (title, abstract)
            pui = item.find(".//*[@idtype='PUI']").text

            # Extract text input, e.g. title, abstract, keywords
            title = "".join(item.find(".//ns0:titletext", prefix_map).itertext())

            try:
                keywords = item.findall(".//ns0:author-keywords//ns0:author-keyword", prefix_map)
                keywords = " ".join([k.text for k in keywords])
            except:
                keywords = ""

            abstracts = item.findall(".//ns0:abstracts//ns1:para", prefix_map)
            abs_1 = "".join(abstracts[0].itertext()) if len(abstracts) != 0 else None
            abs_2 = "".join(abstracts[1].itertext()) if len(abstracts) > 1 else None

            # authors
            authors = \
                [
                    {
                        "initials": author.find('ce:initials', prefix_map).text if author.find('ce:initials', prefix_map) is not None else None,
                        "indexed-name": author.find('ce:indexed-name', prefix_map).text if author.find('ce:indexed-name', prefix_map) is not None else None,
                        "surname": author.find('ce:surname', prefix_map).text if author.find('ce:surname', prefix_map) is not None else None,
                        "given-name": author.find('ce:given-name', prefix_map).text if author.find('ce:given-name', prefix_map) is not None else None,

                    }
                    for author in item.findall(".//ns0:author-group//ns0:author", prefix_map)
                ]

            # Get labels, e.g. check tags/subheadings (_m - manual, _a - automated)
            terms_med_m = item.findall(".//ns0:descriptors[@type='MED']//ns0:mainterm", prefix_map)
            terms_drg_m = item.findall(".//ns0:descriptors[@type='DRG']//ns0:mainterm", prefix_map)
            tags_m = [t.text for t in terms_med_m + terms_drg_m if t.text in clf_labels]

            # For subheadings, the terms are in <links>. Links are only provided by human anotators,
            # therefore we do not have to specify sections.
            links_m = item.findall(".//ns0:descriptors//ns0:link", prefix_map)
            sublinks_m = item.findall(".//ns0:descriptors//ns0:sublink", prefix_map)
            tags_m.extend([l.text for l in links_m + sublinks_m if l.text in clf_labels])

            # Quick test that DRA does not contain links
            links_mea_a = item.findall(".//ns0:descriptors[@type='MEA']//ns0:link", prefix_map)
            links_dra_a = item.findall(".//ns0:descriptors[@type='DRA']//ns0:link", prefix_map)
            assert len(list(links_mea_a + links_dra_a)) == 0, "Subheadings found in MedScan labels!"

            # ---
            # MedScan annotations are not used as labels, only to check.
            terms_a = item.findall(".//ns0:descriptors[@type='MEA']//ns0:mainterm", prefix_map)
            tags_a = ",".join([t.text for t in terms_a if t.text in clf_labels])
            # ---

            # Manual check tags as labels
            labels = list(np.zeros(len(clf_labels), dtype=int))  # include 9 if cdoing check tag groups
            for tag in set(tags_m):
                _idx = clf_labels.index(tag)
                # _idx = check_tags[check_tags.iloc[:, 0] == tag][1].values[0] # include this row if doing check tag groups
                labels[_idx] = 1

            tags_m = ",".join([t for t in tags_m])

            # Append data and one hot encoded manual check tags
            data.append([file, pui, title, keywords, abs_1, abs_2, authors, tags_m, tags_a] + labels)

        try:
            df = pd.concat([df, pd.DataFrame(data, columns=columns)], ignore_index=True)
        except:
            df = pd.DataFrame(data, columns=columns)

        break
        print(f"File {file} done.")

    # Save the DF
    df.drop_duplicates(subset=["pui"], inplace=True)
    df.to_csv(path + "all_articles.csv", index=False, header=True)
    print("Data saved.")


def get_pui_doi_map(path="../../data/", version="A"):
    """
        Function to extract all the pui-doi pars into a CSV file.
    """

    # Load XMLs and extract PUIs
    files = sorted(glob.glob(path + "*.xml"))

    # Define XML namespaces
    prefix_map = {"ns0": "http://www.elsevier.com/xml/ani/ani",
                  "ns1": "http://www.elsevier.com/xml/ani/common"}

    # Define column names
    columns = ["pui", "doi"]

    # Iterate through all files and documents
    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()

        data = []
        for item in root:

            # Get identifiers and input text (title, abstract)
            pui = item.find(".//*[@idtype='PUI']").text
            try:
                doi = item.find(".//ns1:doi", prefix_map).text
            except:
                doi = None
            try:
                pii = item.find(".//ns1:pii", prefix_map).text
            except:
                pii = None

            # Append data and one hot encoded manual check tags
            data.append([pui, doi])

        try:
            df = pd.concat([df, pd.DataFrame(data, columns=columns)], ignore_index=True)
        except:
            df = pd.DataFrame(data, columns=columns)

        print(f"File {file} done.")

    # Save the DF
    df.drop_duplicates(subset=["pui"], inplace=True)
    df.to_csv(path + f"set_{version}_pui_doi.csv", index=False, header=True)
    print("Data saved.")


if __name__ == "__main__":
    parse_document_classification('../data/raw/canary/original_xml_files/')