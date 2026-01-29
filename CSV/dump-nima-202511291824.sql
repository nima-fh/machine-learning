/*M!999999\- enable the sandbox mode */ 
-- MariaDB dump 10.19-12.0.2-MariaDB, for Win64 (AMD64)
--
-- Host: localhost    Database: nima
-- ------------------------------------------------------
-- Server version	12.0.2-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*M!100616 SET @OLD_NOTE_VERBOSITY=@@NOTE_VERBOSITY, NOTE_VERBOSITY=0 */;

--
-- Table structure for table `comments`
--

DROP TABLE IF EXISTS `comments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `comments` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `User` varchar(100) NOT NULL,
  `comment_text` text DEFAULT NULL,
  `Created_at` datetime NOT NULL DEFAULT current_timestamp(),
  `updated_at` datetime DEFAULT NULL,
  `is_Edited` tinyint(1) NOT NULL DEFAULT 0,
  `user-Email` varchar(100) NOT NULL,
  `document_id` int(11) NOT NULL,
  PRIMARY KEY (`ID`),
  KEY `comments_documents_FK` (`document_id`),
  CONSTRAINT `comments_documents_FK` FOREIGN KEY (`document_id`) REFERENCES `documents` (`ID`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=41 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `comments`
--

LOCK TABLES `comments` WRITE;
/*!40000 ALTER TABLE `comments` DISABLE KEYS */;
set autocommit=0;
INSERT INTO `comments` VALUES
(1,'alex_92','Great tutorial, helped me a lot!','2025-11-01 10:15:00','2025-11-01 10:15:00',0,'alex92@example.com',1),
(2,'sara_dev','Can you explain more about joins?','2025-11-02 14:22:00','2025-11-02 14:30:00',1,'sara.dev@example.com',1),
(3,'johnny','This was confusing at first but now clear.','2025-11-03 09:05:00','2025-11-03 09:05:00',0,'johnny@example.com',1),
(4,'maria_k','Loved the visuals in this guide!','2025-11-04 16:45:00','2025-11-04 16:45:00',0,'maria.k@example.com',1),
(5,'dev_dan','I think you missed an edge case.','2025-11-05 11:11:00','2025-11-05 11:20:00',1,'dev.dan@example.com',1),
(6,'lisa88','Thanks for sharing, very useful.','2025-11-06 08:30:00','2025-11-06 08:30:00',0,'lisa88@example.com',1),
(7,'tommy_js','Could you add more SQL examples?','2025-11-07 19:40:00','2025-11-07 19:50:00',1,'tommy.js@example.com',1),
(8,'nina_p','This solved my problem instantly!','2025-11-08 12:00:00','2025-11-08 12:00:00',0,'nina.p@example.com',1),
(9,'coder_mike','I like the step-by-step breakdown.','2025-11-09 17:25:00','2025-11-09 17:25:00',0,'coder.mike@example.com',1),
(10,'emma_r','Could you cover indexes next?','2025-11-10 21:10:00','2025-11-10 21:15:00',1,'emma.r@example.com',1),
(11,'steve_o','Very clear explanation, thanks!','2025-11-11 07:55:00','2025-11-11 07:55:00',0,'steve.o@example.com',1),
(12,'kate_dev','I had trouble replicating this locally.','2025-11-12 13:33:00','2025-11-12 13:40:00',1,'kate.dev@example.com',1),
(13,'ronald','Simple and effective guide.','2025-11-13 09:09:00','2025-11-13 09:09:00',0,'ronald@example.com',1),
(14,'jessy_q','Could you add a diagram?','2025-11-14 15:45:00','2025-11-14 15:50:00',1,'jessy.q@example.com',1),
(15,'paul_dev','This matches what I needed exactly.','2025-11-15 18:20:00','2025-11-15 18:20:00',0,'paul.dev@example.com',1),
(16,'claire_b','I spotted a typo in the query.','2025-11-16 20:05:00','2025-11-16 20:10:00',1,'claire.b@example.com',1),
(17,'leo_x','Awesome explanation, keep it up!','2025-11-17 11:11:00','2025-11-17 11:11:00',0,'leo.x@example.com',1),
(18,'sophie','Could you share a PDF version?','2025-11-18 14:14:00','2025-11-18 14:20:00',1,'sophie@example.com',1),
(19,'daniel_r','This clarified normalization for me.','2025-11-19 09:30:00','2025-11-19 09:30:00',0,'daniel.r@example.com',1),
(20,'olivia_m','Perfect timing, I needed this today.','2025-11-20 22:22:00','2025-11-20 22:22:00',0,'olivia.m@example.com',1);
/*!40000 ALTER TABLE `comments` ENABLE KEYS */;
UNLOCK TABLES;
commit;

--
-- Table structure for table `documents`
--

DROP TABLE IF EXISTS `documents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `documents` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `Subject` varchar(200) NOT NULL,
  `Description` text NOT NULL,
  `Author` varchar(100) DEFAULT NULL,
  `Release Date` datetime DEFAULT current_timestamp(),
  `Last Update` datetime DEFAULT NULL,
  `content` text DEFAULT NULL,
  `Img` text CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `views` int(11) DEFAULT 0,
  `Comments` int(11) DEFAULT 0,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `documents`
--

LOCK TABLES `documents` WRITE;
/*!40000 ALTER TABLE `documents` DISABLE KEYS */;
set autocommit=0;
INSERT INTO `documents` VALUES
(1,'Binary code','Binary code','Nima fakhimi','2025-11-22 11:35:46',NULL,'aadaafdwafwaf',NULL,0,0);
/*!40000 ALTER TABLE `documents` ENABLE KEYS */;
UNLOCK TABLES;
commit;

--
-- Table structure for table `user`
--

DROP TABLE IF EXISTS `user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8mb4 */;
CREATE TABLE `user` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `Name` varchar(100) NOT NULL DEFAULT '',
  `Email` varchar(100) NOT NULL,
  `Password` varchar(100) NOT NULL DEFAULT '',
  PRIMARY KEY (`ID`),
  UNIQUE KEY `user_Email_IDX` (`Email`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user`
--

LOCK TABLES `user` WRITE;
/*!40000 ALTER TABLE `user` DISABLE KEYS */;
set autocommit=0;
INSERT INTO `user` VALUES
(2,'tina','t8713fakhimi@gmail.com','t8713fakhimi');
/*!40000 ALTER TABLE `user` ENABLE KEYS */;
UNLOCK TABLES;
commit;

--
-- Dumping routines for database 'nima'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*M!100616 SET NOTE_VERBOSITY=@OLD_NOTE_VERBOSITY */;

-- Dump completed on 2025-11-29 18:24:30
